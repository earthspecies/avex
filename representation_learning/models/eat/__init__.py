# Re-export the main EAT audio model so callers can simply do
# ``from representation_learning.models.eat import Model``.

from .audio_model import Model  # noqa: F401

__all__ = ["Model"]
