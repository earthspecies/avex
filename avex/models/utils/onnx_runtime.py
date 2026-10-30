"""Shared ONNX Runtime plumbing for the ONNX model wrappers (BirdNet, Perch v2).

These wrappers run a fixed graph through ONNX Runtime, outside PyTorch's
autograd, so they share the same limits: no fine-tuning, and GPU inference
only with the ``onnxruntime-gpu`` build. This module keeps the provider
selection and the user-facing warnings about those limits in one place.
"""

from __future__ import annotations

import logging
from types import ModuleType

import torch

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "uv pip install onnxruntime (CPU) or uv pip install 'onnxruntime-gpu>=1.27' (CUDA 13). "
    "Install one of them, not both (see docs/supported_models.md)."
)


def import_onnxruntime() -> ModuleType:
    """Import ``onnxruntime``, with an install hint naming both builds on failure.

    Returns:
        ModuleType: The ``onnxruntime`` module.

    Raises:
        ImportError: If neither ``onnxruntime`` nor ``onnxruntime-gpu`` is installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"onnxruntime is required for this model: {_INSTALL_HINT}") from e
    return ort


def ort_providers() -> list[str]:
    """Return CUDA then CPU execution providers, keeping only those this build has.

    Requesting ``CUDAExecutionProvider`` on the CPU build makes ONNX Runtime
    log a warning on every session; filtering avoids that.

    Returns:
        list[str]: The providers to pass to ``InferenceSession``.
    """
    available = import_onnxruntime().get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def warn_if_cpu_fallback(model_name: str, device: str | torch.device, providers: list[str]) -> None:
    """Warn when a CUDA device was requested but the session runs on CPU.

    Args:
        model_name: Name used in the message.
        device: The device the user asked for.
        providers: ``session.get_providers()`` of the live session.
    """
    if str(device).startswith("cuda") and "CUDAExecutionProvider" not in providers:
        logger.warning(
            "%s runs on CPU (ONNX Runtime providers: %s); outputs are copied to %s. "
            "For GPU inference replace onnxruntime with onnxruntime-gpu>=1.27 (see docs/supported_models.md).",
            model_name,
            providers,
            device,
        )


def warn_cannot_unfreeze(model: torch.nn.Module) -> None:
    """Warn, once per model instance, that ``freeze_backbone=False`` has no effect.

    Args:
        model: The wrapper; a flag is stored on it so the warning fires once.
    """
    if getattr(model, "_warned_cannot_unfreeze", False):
        return
    model._warned_cannot_unfreeze = True
    logger.warning(
        "freeze_backbone=False has no effect on %s: ONNX Runtime runs outside autograd, "
        "so the backbone cannot be fine-tuned. Only layers on top of it train.",
        type(model).__name__,
    )
