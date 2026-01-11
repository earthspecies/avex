"""Model factory for representation learning models. (Deprecated)

This module provides a legacy factory function to instantiate different types of
representation learning models based on configuration specifications.

Deprecated:
    Use ``representation_learning.models.utils.factory.build_model_from_spec``
    instead. The registry-based factory is the supported, extensible API for
    creating models from ``ModelSpec``.
"""

import logging

from representation_learning.configs import ModelSpec
from representation_learning.models.aves_model import Model as AVESModel
from representation_learning.models.base_model import ModelBase
from representation_learning.models.clip import CLIPModel
from representation_learning.models.efficientnet import (
    Model as EfficientNet,
)
from representation_learning.models.perch import Model as PerchModel
from representation_learning.models.resnet import Model as ResNetModel

logger = logging.getLogger(__name__)


def get_model(model_config: ModelSpec, num_classes: int) -> ModelBase:
    """
    Factory function to obtain a model instance based on a static list of supported
    models.
    Model implementations are expected to reside in their own modules (e.g.
    efficientnet.py) and define a class (always called 'Model'). This function
    currently supports:
    - 'efficientnet': Audio classification model
    - 'clip': CLIP-like model for audio-text contrastive learning
    - 'perch': Google's Perch bird audio classification model
    - 'atst': ATST Frame model for timestamp embeddings
    - 'birdmae': Bird-MAE pretrained model for bird audio classification
    - 'biolingual': BioLingual zero-shot audio classification model

    Args:
        model_config: Model configuration object containing:
            - name: Name of the model to instantiate
            - pretrained: Whether to use pretrained weights
            - device: Device to run on
            - audio_config: Audio processing configuration
            - text_model_name: (for CLIP) Name of the text model to use
            - projection_dim: (for CLIP) Dimension of the projection space
            - temperature: (for CLIP) Temperature for contrastive loss
            - atst_model_path: (for ATST) Path to pretrained ATST model checkpoint

    Returns:
        An instance of the corresponding model configured with the provided
        parameters.
    """
    logger.warning(
        "get_model() is deprecated and will be removed in a future release. "
        "Use representation_learning.models.utils.factory.build_model_from_spec() instead."
    )

    model_name = model_config.name.lower()

    if model_name == "efficientnet":
        return EfficientNet(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            efficientnet_variant=getattr(model_config, "efficientnet_variant", "b0"),
        )
    elif model_name == "aves_bio":
        return AVESModel(
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
            efficientnet_variant=getattr(model_config, "efficientnet_variant", "b0"),
        )
    elif model_name == "perch":
        return PerchModel(
            num_classes=num_classes,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )
    elif model_name == "atst":
        from representation_learning.models.atst_frame.atst_encoder import (
            Model as ATSTModel,
        )

        # ATST requires model path
        atst_model_path = getattr(model_config, "atst_model_path", "pretrained/atst_as2M.pt")
        # atst_model_path = "pretrained/atst_frame_base.pt"

        return ATSTModel(
            atst_model_path=atst_model_path,
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )
    elif model_name in {"resnet18", "resnet50", "resnet152"}:
        return ResNetModel(
            variant=model_name,
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )

    elif model_name == "beats":
        from representation_learning.models.beats_model import (
            Model as BeatsModel,
        )

        use_naturelm = getattr(model_config, "use_naturelm", False)
        fine_tuned = getattr(model_config, "fine_tuned", False)
        disable_layerdrop = getattr(model_config, "disable_layerdrop", False)
        beats_variant = getattr(model_config, "beats_variant", None)
        openbeats_size = getattr(model_config, "openbeats_size", "base")

        return BeatsModel(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            use_naturelm=use_naturelm,
            fine_tuned=fine_tuned,
            disable_layerdrop=disable_layerdrop,
            beats_variant=beats_variant,
            openbeats_size=openbeats_size,
        )
    elif model_name == "eat_hf":
        from representation_learning.models.eat_hf import (
            Model as EATHFModel,  # Local import to avoid HF deps when unused
        )

        target_length = getattr(model_config, "target_length", 1024)
        pooling = getattr(model_config, "pooling", "cls")
        model_id = model_config.model_id or "worstchan/EAT-base_epoch30_pretrain"
        fairseq_weights_path = getattr(model_config, "fairseq_weights_path", None)
        norm_mean = getattr(model_config, "eat_norm_mean", -4.268)
        norm_std = getattr(model_config, "eat_norm_std", 4.569)

        return EATHFModel(
            model_name=model_id,
            num_classes=num_classes,
            device=model_config.device,
            audio_config=model_config.audio_config,
            target_length=target_length,
            pooling=pooling,
            fairseq_weights_path=fairseq_weights_path,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    elif model_name == "birdnet":
        from representation_learning.models.birdnet import (
            Model as BirdNetModel,  # Local import to avoid TF deps when unused
        )

        return BirdNetModel(
            num_classes=num_classes,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )
    elif model_name == "birdmae":
        from representation_learning.models.birdmae import (
            Model as BirdMAEModel,  # Local import to avoid transformers deps
        )

        model_id = getattr(model_config, "model_id", "DBD-research-group/Bird-MAE-Base")

        return BirdMAEModel(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            model_id=model_id,
        )
