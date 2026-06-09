"""
optimizer.py — Adam optimizer with Noam (Transformer) learning rate schedule.

THE SCHEDULE FROM THE PAPER (Section 5.3):
-----------------------------------------
lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})

This schedule has two phases:
  PHASE 1 (step ≤ warmup_steps): LINEAR WARMUP
    The active term is: step_num * warmup_steps^{-1.5}
    This grows linearly from ~0 to peak. Peak is at step = warmup_steps.

  PHASE 2 (step > warmup_steps): INVERSE SQRT DECAY
    The active term is: step_num^{-0.5}. This decays as 1/sqrt(step), slowly decreasing.

Warmup gives the optimizer time to:
  1. Build up reliable gradient statistics
  2. Allow the model to find a reasonable initial direction before large steps

WHY INVERSE SQRT DECAY?
-----------------------
As training progresses, the model gets closer to a good solution. We want smaller updates to "fine-tune" rather than overshoot. 1/sqrt(step) is
a common schedule - it's slow enough to keep making progress, but decreasing enough to converge.

OPTIMIZER HYPERPARAMETERS:
  beta_1 = 0.9   (momentum: exponential decay of past gradients)
  beta_2 = 0.98  (RMSProp: exponential decay of past squared gradients)
  e = 1e-9    (numerical stability; very small to let Adam's learning rate schedule dominate)

"""

import torch
import torch.optim as optim
from config import cfg


class NoamScheduler:
    """
    The Noam learning rate scheduler from "Attention Is All You Need."

    This is NOT a PyTorch LRScheduler subclass - it manually sets the
    learning rate at each step. This is simpler and more transparent.

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

        for batch in data:
            loss = ...
            loss.backward()
            optimizer.step()
            scheduler.step()   <- call AFTER optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int = cfg.d_model,
        warmup_steps: int = cfg.warmup_steps,
        factor: float = 1.0
    ):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer, The optimizer whose learning rate we modify.
            Should be initialized with lr=1.0 (we'll override it every step).
        d_model : int, Model dimension. Appears in the formula as d_model^{-0.5}.
        warmup_steps : int, Number of warmup steps. Paper: 4000.
        factor : float Scaling factor.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0      # current step number (0-indexed, incremented before use)
        self._rate = 0.0    # current learning rate (for logging)

    def step(self):
        """
        Increment step counter and update learning rate. Call this AFTER optimizer.step() each training iteration.
        """
        self._step += 1
        rate = self._compute_lr(self._step)
        self._rate = rate

        # Set the learning rate in all parameter groups of the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate

    def _compute_lr(self, step: int) -> float:
        """
        Compute the learning rate for a given step.
        Formula:
          lr = factor * d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

        Parameters
        ----------
        step : int, Current training step (1-indexed).

        Returns
        -------
        lr : float, Learning rate for this step.
        """
        # Avoid step=0 (division by zero / undefined)
        step = max(step, 1)

        # d_model^{-0.5}
        d_model_factor = self.d_model ** (-0.5)

        # min(step^{-0.5}, step * warmup^{-1.5})
        arg1 = step ** (-0.5)                         # decay term
        arg2 = step * (self.warmup_steps ** (-1.5))   # warmup term

        return self.factor * d_model_factor * min(arg1, arg2)

    @property
    def current_lr(self) -> float:
        """Return the current learning rate."""
        return self._rate

    @property
    def current_step(self) -> int:
        """Return the current step number."""
        return self._step

    def state_dict(self) -> dict:
        """Save scheduler state for checkpointing."""
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            '_step': self._step,
            '_rate': self._rate,
        }

    def load_state_dict(self, state: dict):
        """Restore scheduler state from checkpoint."""
        self._step = state['_step']
        self._rate = state['_rate']
        # Apply the loaded LR to optimizer immediately
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._rate


def build_optimizer(model: torch.nn.Module, config=cfg):
    """
    Build the Adam optimizer and Noam LR scheduler from config.

    Returns
    -------
    optimizer : torch.optim.Adam
    scheduler : NoamScheduler
    """
    # Adam with paper's hyperparameters. # lr=1.0 is a placeholder - NoamScheduler overwrites it every step.
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0,         # overridden by scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps,
        factor=0.5        # changed from 1.0 to 0.5
    )

    return optimizer, scheduler
