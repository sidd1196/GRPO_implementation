import os
import sys
from typing import Callable
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import softmax
from einops import einsum, rearrange, repeat
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_reward(prompts: torch.Tensor, responses: torch.Tensor, reward_fn: Callable[[list[int], list[int]], float]) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        rewards (float[batch trial])
    """
    batch_size, num_responses, _ = responses.shape
    rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(num_responses):
            rewards[i, j] = reward_fn(prompts[i, :].tolist(), responses[i, j, :].tolist())
    return rewards


def sort_distance_reward(prompt: list[int], response: list[int]) -> float:
    """
    Return how close response is to ground_truth = sorted(prompt).
    In particular, compute number of positions where the response matches the ground truth.
    """
    assert len(prompt) == len(response)
    ground_truth = sorted(prompt)
    return sum(1 for x, y in zip(response, ground_truth) if x == y)


def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:
    """
    Return how close response is to ground_truth = sorted(prompt).
    """
    assert len(prompt) == len(response)

    # Give one point for each token in the prompt that shows up in the response
    inclusion_reward = sum(1 for x in prompt if x in response)

    # Give one point for each adjacent pair in response that's sorted
    ordering_reward = sum(1 for x, y in zip(response, response[1:]) if x <= y)

    return inclusion_reward + ordering_reward


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, prompt_length: int, response_length: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # For each position, we have a matrix for encoding and a matrix for decoding
        self.encode_weights = nn.Parameter(torch.randn(prompt_length, embedding_dim, embedding_dim) /
                                          math.sqrt(embedding_dim))
        self.decode_weights = nn.Parameter(torch.randn(response_length, embedding_dim, embedding_dim) /
                                          math.sqrt(embedding_dim))

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: int[batch pos]
        Returns:
            logits: float[batch pos vocab]
        """
        # Embed the prompts
        embeddings = self.embedding(prompts)  # [batch pos dim]

        # Transform using per prompt position matrix, collapse into one vector
        encoded = einsum(embeddings, self.encode_weights, "batch pos dim1, pos dim1 dim2 -> batch dim2")

        # Turn into one vector per response position
        decoded = einsum(encoded, self.decode_weights, "batch dim2, pos dim2 dim1 -> batch pos dim1")

        # Convert to logits (input and output share embeddings)
        logits = einsum(decoded, self.embedding.weight, "batch pos dim1, vocab dim1 -> batch pos vocab")

        return logits

    def clone(self):
        """Create a deep copy of the model."""
        cloned = Model(
            vocab_size=self.embedding.num_embeddings,
            embedding_dim=self.embedding_dim,
            prompt_length=self.encode_weights.shape[0],
            response_length=self.decode_weights.shape[0]
        )
        cloned.load_state_dict(self.state_dict())
        return cloned


def generate_responses(prompts: torch.Tensor, model: Model, num_responses: int) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
    Returns:
        generated responses: int[batch trial pos]

    Example (batch_size = 3, prompt_length = 3, num_responses = 2, response_length = 4)
        p1 p1 p1 r1 r1 r1 r1
        r2 r2 r2 r2
        p2 p2 p2 r3 r3 r3 r3
        r4 r4 r4 r4
        p3 p3 p3 r5 r5 r5 r5
        r6 r6 r6 r6
    """
    logits = model(prompts)  # [batch pos vocab]
    batch_size = prompts.shape[0]

    # Sample num_responses (independently) for each [batch pos]
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    flattened_responses = torch.multinomial(softmax(flattened_logits, dim=-1), num_samples=num_responses,
                                          replacement=True)  # [batch pos trial]
    responses = rearrange(flattened_responses, "(batch pos) trial -> batch trial pos", batch=batch_size)
    return responses


def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, model: Model) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        log_probs (float[batch trial pos]) under the model
    """
    # Compute log prob of responses under model
    logits = model(prompts)  # [batch pos vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch pos vocab]

    # Replicate to align with responses
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", trial=num_responses)  # [batch trial pos vocab]

    # Index into log_probs using responses
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)  # [batch trial pos]

    return log_probs


def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) which are advantage-like quantities for updating
    """
    if mode == "rewards":
        return rewards

    if mode == "centered_rewards":
        # Compute mean over all the responses (trial) for each prompt (batch)
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        centered_rewards = rewards - mean_rewards
        return centered_rewards

    if mode == "normalized_rewards":
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        std_rewards = rewards.std(dim=-1, keepdim=True)
        centered_rewards = rewards - mean_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)
        return normalized_rewards

    if mode == "max_rewards":
        # Zero out any reward that isn't the maximum for each batch
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards

    raise ValueError(f"Unknown mode: {mode}")


def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "unclipped":
        ratios = torch.exp(log_probs - old_log_probs)  # [batch trial pos]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = torch.exp(log_probs - old_log_probs)  # [batch trial pos]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")

        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()

    raise ValueError(f"Unknown mode: {mode}")


def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute an estimate of KL(model | ref_model), where the models are given by:
    log_probs [batch trial pos]
    ref_log_probs [batch trial pos]
    Use the estimate:
    KL(p || q) = E_p[q/p - log(q/p) - 1]
    """
    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()


def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        compute_ref_model_period: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "rewards",
                        loss_mode: str = "naive",
                        kl_penalty: float = 0.0,
                        reward_fn: Callable[[list[int], list[int]], float] = sort_inclusion_ordering_reward,
                        use_cache: bool = False,
                        device: str = "cpu") -> tuple[str, str]:
    """Train a model using policy gradient.
    Return:
        - Path to the image of the learning curve.
        - Path to the log file
    """
    torch.manual_seed(5)

    # Create output directory if it doesn't exist
    os.makedirs("var", exist_ok=True)

    image_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.png"
    log_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.txt"

    # Already ran, just cache it
    if use_cache and os.path.exists(image_path) and os.path.exists(log_path):
        return image_path, log_path

    # Define the data
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]], device=device)
    vocab_size = prompts.max().item() + 1
    prompt_length = response_length = prompts.shape[1]

    model = Model(vocab_size=vocab_size, embedding_dim=10, prompt_length=prompt_length, response_length=response_length)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    records = []
    ref_log_probs = None
    ref_model = None
    old_log_probs = None

    if use_cache:
        out = open(log_path, "w")
    else:
        out = sys.stdout

    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # If using KL penalty, need to get the reference model (freeze it every few epochs)
        if kl_penalty != 0:
            if epoch % compute_ref_model_period == 0:
                ref_model = model.clone()

        # Sample responses and evaluate their rewards
        responses = generate_responses(prompts=prompts, model=model, num_responses=num_responses)  # [batch trial pos]
        rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=reward_fn)  # [batch trial]
        deltas = compute_deltas(rewards=rewards, mode=deltas_mode)  # [batch trial]

        if kl_penalty != 0:  # Compute under the reference model
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=ref_model)  # [batch trial pos]

        if loss_mode != "naive":  # Compute under the current model (but freeze while we do the inner steps)
            with torch.no_grad():
                old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial pos]

        # Take a number of steps given the responses
        for step in range(num_steps_per_epoch):
            log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial pos]
            loss = compute_loss(log_probs=log_probs, deltas=deltas, mode=loss_mode, old_log_probs=old_log_probs)
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs=log_probs, ref_log_probs=ref_log_probs)

            # Print information
            print_information(epoch=epoch, step=step, loss=loss, prompts=prompts, rewards=rewards, responses=responses,
                            log_probs=log_probs, deltas=deltas, out=out)

            global_step = epoch * num_steps_per_epoch + step
            records.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "mean_reward":
                          rewards.mean().item()})

            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if use_cache:
        out.close()

    if use_cache:
        # Plot step versus loss and reward in two subplots
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        rewards = [r["mean_reward"] for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss subplot
        ax1.plot(steps, losses)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Train Loss")

        # Reward subplot
        ax2.plot(steps, rewards)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Mean Reward")
        ax2.set_title("Mean Reward")

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

    return image_path, log_path


def print_information(epoch: int, step: int, loss: torch.Tensor, prompts: torch.Tensor, rewards: torch.Tensor, responses:
                      torch.Tensor, log_probs: torch.Tensor, deltas: torch.Tensor, out):
    print(f"epoch = {epoch}, step = {step}, loss = {loss:.3f}, reward = {rewards.mean():.3f}", file=out)
    if epoch % 1 == 0 and step % 5 == 0:
        for batch in range(prompts.shape[0]):
            print(f"  prompt = {prompts[batch, :].tolist()}", file=out)
            for trial in range(responses.shape[1]):
                print(f"  response = {responses[batch, trial, :].tolist()}, log_probs = {tstr(log_probs[batch, trial])}, reward = {rewards[batch, trial]:.3f}, delta = {deltas[batch, trial]:.3f}", file=out)


def tstr(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x[i]:.3f}" for i in range(x.shape[0])) + "]"


def experiments():
    """Run experiments with different delta modes."""
    print("Running experiments with different delta modes...")
    
    # Let's start with updating based on raw rewards.
    print("\n1. Raw rewards mode:")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10,
                                              deltas_mode="rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)
    print(f"Results saved to: {image_path}, {log_path}")

    # Let's try using centered rewards.
    print("\n2. Centered rewards mode:")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10,
                                              deltas_mode="centered_rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)
    print(f"Results saved to: {image_path}, {log_path}")

    # Finally, let's try normalizing by the standard deviation.
    print("\n3. Normalized rewards mode:")
    image_path, log_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10,
                                              deltas_mode="normalized_rewards", loss_mode="naive", reward_fn=sort_inclusion_ordering_reward, use_cache=True)
    print(f"Results saved to: {image_path}, {log_path}")


if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run experiments
    experiments()
