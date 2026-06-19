"""Custom TrainerCallbacks — the place to extend TRL without forking it.

TwoStageTemperatureCallback implements MicroCoder-GRPO Fix 2 (diversity-
determined temperature) as a schedule: a low, stable temperature early to keep
gradients clean, then a higher temperature after `switch_step` to widen
exploration once the policy has stabilised. TRL has no native temperature
schedule, so we mutate the trainer's sampling temperature in-place at step
boundaries — which is exactly the kind of hook these callbacks exist for.
"""

from __future__ import annotations

from transformers import TrainerCallback


class TwoStageTemperatureCallback(TrainerCallback):
    def __init__(self, switch_step: int, temp_stage1: float, temp_stage2: float):
        self.switch_step = switch_step
        self.temp_stage1 = temp_stage1
        self.temp_stage2 = temp_stage2
        self._switched = False
        self.trainer = None  # injected in train.py after the trainer is built

    def _set_temp(self, trainer, value: float):
        # GRPOTrainer reads self.temperature when building the generation /
        # vLLM SamplingParams for each rollout. Set both the attribute and the
        # args mirror so it sticks regardless of which one the version reads.
        if hasattr(trainer, "temperature"):
            trainer.temperature = value
        if hasattr(trainer.args, "temperature"):
            trainer.args.temperature = value

    def on_step_begin(self, args, state, control, **kwargs):
        if self._switched or state.global_step < self.switch_step:
            return
        self._set_temp(self.trainer, self.temp_stage2)
        self._switched = True
        print(
            f"[TwoStageTemperature] step {state.global_step}: "
            f"temperature {self.temp_stage1} -> {self.temp_stage2}"
        )
