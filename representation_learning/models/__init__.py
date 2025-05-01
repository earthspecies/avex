from typing import Dict, Type

from representation_learning.configs import ModelSpec
from representation_learning.models.clip import CLIPModel
from representation_learning.models.efficientnetb0 import EfficientNetB0

# Map model names to their classes
MODEL_REGISTRY: Dict[str, Type] = {
    "efficientnetb0": EfficientNetB0,
    "clip": CLIPModel,
}


def get_model(cfg: ModelSpec) -> EfficientNetB0 | CLIPModel:
    """Get a model instance based on configuration.

    Parameters
    ----------
    cfg : ModelSpec
        Model configuration

    Returns
    -------
    EfficientNetB0 | CLIPModel
        Model instance

    Raises
    ------
    ValueError
        If model name is unknown
    """
    if cfg.name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {cfg.name}")
    return MODEL_REGISTRY[cfg.name]()
