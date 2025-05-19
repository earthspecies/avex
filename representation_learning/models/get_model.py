from representation_learning.configs import ModelSpec
from representation_learning.models.base_model import ModelBase
from representation_learning.models.clip import CLIPModel
from representation_learning.models.efficientnetb0 import (
    Model as EfficientNetB0,
)
from representation_learning.models.resnet import Model as ResNetModel


def get_model(model_config: ModelSpec, num_classes: int) -> ModelBase:
    """
    Factory function to obtain a model instance based on a static list of supported
    models.
    Model implementations are expected to reside in their own modules (e.g.
    efficientnetb0.py) and define a class (always called 'Model'). This function
    currently supports:
    - 'efficientnetb0': Audio classification model
    - 'clip': CLIP-like model for audio-text contrastive learning

    Args:
        model_config: Model configuration object containing:
            - name: Name of the model to instantiate
            - pretrained: Whether to use pretrained weights
            - device: Device to run on
            - audio_config: Audio processing configuration
            - text_model_name: (for CLIP) Name of the text model to use
            - projection_dim: (for CLIP) Dimension of the projection space
            - temperature: (for CLIP) Temperature for contrastive loss
        num_classes: The number of classes to be used in the model.

    Returns:
        An instance of the corresponding model configured with the provided
        parameters.

    Raises:
        NotImplementedError: If the model_config does not match any supported
        models.
    """
    model_name = model_config.name.lower()

    if model_name == "efficientnetb0":
        return EfficientNetB0(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )

    elif model_name == "clip":
        return CLIPModel(
            device=model_config.device,
            audio_config=model_config.audio_config,
            text_model_name=getattr(model_config, "text_model_name", "roberta-base"),
            projection_dim=getattr(model_config, "projection_dim", 512),
            temperature=getattr(model_config, "temperature", 0.07),
        )
    elif model_name in {"resnet18", "resnet50", "resnet152"}:
        return ResNetModel(
            variant=model_name,
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )
    else:
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Supported models: "
            "'efficientnetb0', 'clip', 'resnet18', 'resnet50', 'resnet152'."
        )
