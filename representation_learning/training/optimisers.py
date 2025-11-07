"""
Utility for constructing PyTorch optimisers from the `training_params` section
of a run configuration.\
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer

from representation_learning.configs import TrainingParams

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def get_optimizer(
    params: Iterable[torch.nn.parameter.Parameter],
    training_params: TrainingParams,
) -> Optimizer:
    """Return a *ready‑to‑use* optimiser.

    Parameters
    ----------
    params
        Parameters to optimise – usually ``model.parameters()``.
    training_params
        Either the *sub‑section* of your Pydantic config (which supports
        attribute access) **or** a raw dict.  Must expose at least:

        - ``optimizer`` (str) – one of **adamw**, **adam**.
        - ``lr`` (float)
        - ``weight_decay`` (float)

    Returns
    -------
    torch.optim.Optimizer
        Instantiated optimiser ready for training.

    Raises
    ------
    ValueError
        If an unsupported optimizer name is provided.
    """

    # --------------------------------------------------------------------- #
    #  Normalise inputs

    # --------------------------------------------------------------------- #
    opt_name: str = training_params.optimizer.lower()
    lr: float = float(training_params.lr)
    weight_decay: float = float(training_params.weight_decay)
    adam_betas = getattr(training_params, "adam_betas", None)

    # --------------------------------------------------------------------- #
    #  Parameter grouping – honour `weight_decay_scale` & `param_group`      #
    # --------------------------------------------------------------------- #

    grouped: dict[tuple[float, str], list[torch.nn.Parameter]] = {}
    for p in params:
        if not p.requires_grad:
            continue

        # Default values when no overrides are present
        scale = 1.0
        group_name = "default"

        if hasattr(p, "optim_overrides"):
            opt_over = p.optim_overrides or {}
            scale = opt_over.get("optimizer", {}).get("weight_decay_scale", 1.0)

        if hasattr(p, "param_group"):
            group_name = p.param_group or group_name

        grouped.setdefault((scale, group_name), []).append(p)

    # Build param-group dicts with effective weight-decay = base * scale
    param_groups = []
    for (scale, gname), plist in grouped.items():
        param_groups.append(
            {
                "params": plist,
                "weight_decay": weight_decay * float(scale),
                "lr": lr,
                "group_name": gname,
                **({"betas": tuple(adam_betas)} if adam_betas is not None else {}),
            }
        )

    # --------------------------------------------------------------------- #
    #  Factory
    # --------------------------------------------------------------------- #
    if opt_name == "adamw":
        optimiser_cls = torch.optim.AdamW  # type: ignore[assignment]
    elif opt_name == "adam":
        optimiser_cls = torch.optim.Adam  # type: ignore[assignment]
    elif opt_name == "adamw8bit":
        import bitsandbytes as bnb

        optimiser_cls = bnb.optim.PagedAdamW8bit  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Available: adamw, adam, adamw8bit.")

    return optimiser_cls(param_groups)
