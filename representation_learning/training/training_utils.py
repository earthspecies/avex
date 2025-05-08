"""
representation_learning.training.training_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility functions for training, including scheduler creation.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, _LRScheduler

from representation_learning.configs import RunConfig


def build_scheduler(
    optimizer: Optimizer,
    cfg: RunConfig,
    total_steps: int,
) -> _LRScheduler:
    """Build the learning rate scheduler based on configuration.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer to schedule
    cfg : RunConfig
        Configuration containing scheduler settings
    total_steps : int
        Total number of training steps

    Returns
    -------
    _LRScheduler
        The configured learning rate scheduler

    Raises
    ------
    ValueError
        If the scheduler name is unknown
    """
    if cfg.scheduler.name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Use warmup_steps directly from config
    warmup_steps = cfg.scheduler.warmup_steps

    if cfg.scheduler.name == "cosine":
        # Create warmup scheduler
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        # Create cosine annealing scheduler
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.scheduler.min_lr,
        )
        # Combine schedulers
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    elif cfg.scheduler.name == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=cfg.scheduler.min_lr / cfg.training_params.lr,
            total_iters=total_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")
