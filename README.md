# GRPO (Group Relative Policy Optimization) Implementation

A PyTorch implementation of Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm for training language models. GRPO is a simplification of Proximal Policy Optimization (PPO) that removes the critic/value function and leverages the group structure in language model settings to provide natural baselines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Experiments](#experiments)
- [Results](#results)
- [References](#references)

## Overview

This project implements GRPO for training language models using policy gradient methods. The implementation includes:

- **Policy Gradient Training**: Multiple baseline strategies for variance reduction
- **Reward Functions**: Custom reward functions for sequence tasks (sorting)
- **Neural Network Architecture**: Position-aware encoder-decoder model
- **GPU Support**: CUDA acceleration for faster training
- **Multiple Training Modes**: Naive, unclipped, and clipped loss functions

The task demonstrated here is **number sorting** - training a model to sort sequences of numbers using reinforcement learning with verifiable rewards.

## Features

- **Complete GRPO Implementation**: Full policy gradient algorithm with group-based baselines
- **Multiple Baseline Strategies**:
  - Raw rewards
  - Centered rewards (GRPO baseline)
  - Normalized rewards
  - Max rewards
- **Flexible Loss Functions**:
  - Naive policy gradient
  - Unclipped importance sampling (PPO-style)
  - Clipped importance sampling (PPO-style)
- **Reward Function Engineering**: 
  - Distance-based rewards
  - Inclusion and ordering rewards with partial credit
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Visualization**: Learning curves and training metrics
- **Jupyter Notebook**: Interactive Colab-ready notebook for experimentation

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+ (with CUDA support recommended)
- einops
- matplotlib
- tqdm

### Install Dependencies

```bash
pip install torch einops matplotlib tqdm
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Python Script

Run the complete implementation:

```bash
python grpo_implementation.py
```

This will run all experiments with different delta modes and save results to the `var/` directory.

### Jupyter Notebook (Google Colab)

1. Upload `grpo_implementation.ipynb` to Google Colab
2. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
3. Run all cells sequentially
4. Results will be displayed inline with plots and metrics

### Custom Training

```python
from grpo_implementation import run_policy_gradient, sort_inclusion_ordering_reward
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Run training with custom parameters
image_path, log_path = run_policy_gradient(
    num_epochs=100,
    num_steps_per_epoch=10,
    num_responses=10,
    deltas_mode="centered_rewards",  # Options: "rewards", "centered_rewards", "normalized_rewards", "max_rewards"
    loss_mode="naive",  # Options: "naive", "unclipped", "clipped"
    reward_fn=sort_inclusion_ordering_reward,
    use_cache=True,
    device=device
)
```

## Project Structure

```
GRPO_implementation/
│
├── grpo_implementation.py          # Main Python implementation
├── grpo_implementation.ipynb      # Jupyter notebook for Colab
├── README.md                       # This file
├── code_instrcutons_for_rl.pdf    # Original lecture notes
└── var/                            # Output directory (created automatically)
    ├── policy_gradient_*.png      # Learning curve plots
    └── policy_gradient_*.txt      # Training logs
```

## Technical Details

### Model Architecture

The model uses a position-aware encoder-decoder architecture:

- **Embedding Layer**: Maps vocabulary tokens to dense vectors
- **Positional Encoding**: Separate encoding/decoding weights for each position
- **Non-autoregressive**: Generates all positions independently (simplified for demonstration)

### Policy Gradient Methods

The implementation supports several policy gradient variants:

1. **Naive Policy Gradient**: `∇ log π(a|s) R(s,a)`
2. **Baselined Policy Gradient**: `∇ log π(a|s) (R(s,a) - b(s))`
   - Baseline `b(s)` reduces variance without changing the gradient expectation
3. **GRPO Baseline**: Uses mean reward across multiple responses per prompt as baseline

### Reward Functions

Two reward functions are implemented:

1. **`sort_distance_reward`**: Exact match reward (sparse)
   - Returns number of positions matching ground truth
   
2. **`sort_inclusion_ordering_reward`**: Partial credit reward (dense)
   - Inclusion reward: Points for tokens present in response
   - Ordering reward: Points for correctly ordered adjacent pairs

### Delta Computation Modes

- **`rewards`**: Raw rewards (no baseline)
- **`centered_rewards`**: Rewards minus mean (GRPO baseline)
- **`normalized_rewards`**: Centered and normalized by standard deviation
- **`max_rewards`**: Only maximum reward per batch (winner-take-all)

## Experiments

The implementation includes experiments comparing different baseline strategies:

1. **Raw Rewards**: Baseline policy gradient without variance reduction
2. **Centered Rewards**: GRPO approach using group mean as baseline
3. **Normalized Rewards**: Centered rewards with standard deviation normalization
4. **Clipped Loss**: PPO-style clipped importance sampling ratios

### Running Experiments

```python
# Experiment 1: Raw rewards
run_policy_gradient(deltas_mode="rewards", loss_mode="naive", ...)

# Experiment 2: Centered rewards (GRPO)
run_policy_gradient(deltas_mode="centered_rewards", loss_mode="naive", ...)

# Experiment 3: Normalized rewards
run_policy_gradient(deltas_mode="normalized_rewards", loss_mode="naive", ...)

# Experiment 4: Clipped loss with centered rewards
run_policy_gradient(deltas_mode="centered_rewards", loss_mode="clipped", ...)
```

## Results

Training results are saved to the `var/` directory:

- **Plots**: Learning curves showing loss and mean reward over training steps
- **Logs**: Detailed training logs with prompts, responses, rewards, and deltas

The plots compare:
- Training loss evolution
- Mean reward improvement over time

## Key Concepts

### Policy Gradient

The policy gradient theorem states:
```
∇ E[R] = E[∇ log π(a|s) R(s,a)]
```

### Baselines

Adding a baseline `b(s)` doesn't change the expectation:
```
E[∇ log π(a|s) (R(s,a) - b(s))] = E[∇ log π(a|s) R(s,a)]
```

But reduces variance when `b(s) ≈ E[R|s]`.

### GRPO Innovation

GRPO leverages the natural group structure in language models (multiple responses per prompt) to compute baselines without requiring a separate value function network.

## References

- **GRPO Paper**: Group Relative Policy Optimization (Yang et al., 2025)
- **PPO Paper**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- **Stanford CS336**: Reinforcement Learning from Human Feedback (Spring 2025)
- **Original Lecture**: Based on CS336 Lecture 17 - Policy Gradient Deep Dive

## Contributing

This is an educational implementation. Feel free to:
- Experiment with different reward functions
- Try different model architectures
- Extend to other sequence tasks
- Optimize for larger models

## License

This implementation is for educational purposes. Please refer to the original papers and course materials for licensing information.

## Acknowledgments

- Stanford CS336 course materials
- PyTorch team for excellent deep learning framework
- Google Colab for GPU resources

---

**Note**: This implementation is based on educational materials and is intended for learning purposes. For production use, refer to the official GRPO paper and implementations.
