"""
Utility for constructing PyTorch optimisers from the `training_params` section
of a run configuration.\
"""

from __future__ import annotations

from typing import Iterable, Type

import bitsandbytes as bnb
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

    # --------------------------------------------------------------------- #
    #  Factory
    # --------------------------------------------------------------------- #
    if opt_name == "adamw":
        optimiser_cls: Type[Optimizer] = torch.optim.AdamW
        kwargs = {"lr": lr, "weight_decay": weight_decay}
    elif opt_name == "adam":
        optimiser_cls = torch.optim.Adam
        kwargs = {"lr": lr, "weight_decay": weight_decay}
    elif opt_name == "adamw8bit":
        optimiser_cls: Type[Optimizer] = bnb.optim.PagedAdamW8bit
        kwargs = {"lr": lr, "weight_decay": weight_decay}
    else:
        raise ValueError(
            f"Unsupported optimizer '{opt_name}'. Available: adamw, adam, adamw8bit."
        )

    return optimiser_cls(params, **kwargs)
