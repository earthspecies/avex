from representation_learning.configs import ModelSpec
from representation_learning.models.aves_model import Model as AVESModel
from representation_learning.models.base_model import ModelBase
from representation_learning.models.clip import CLIPModel
from representation_learning.models.dummy_model import Model as DummyModel
from representation_learning.models.efficientnet import (
    Model as EfficientNet,
)
from representation_learning.models.perch import Model as PerchModel
from representation_learning.models.perch_bacpipe import Model as PerchTensorFlowHubModel
from representation_learning.models.resnet import Model as ResNetModel


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
    - 'perch_bacpipe': Google's Perch model via TensorFlow Hub
    - 'atst': ATST Frame model for timestamp embeddings

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

    Raises:
        NotImplementedError: If the model_config does not match any supported
        models.
    """
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
    elif model_name == "perch_bacpipe":
        return PerchTensorFlowHubModel(
            num_classes=num_classes,
            device=model_config.device,
            audio_config=model_config.audio_config,
        )
    elif model_name == "atst":
        from representation_learning.models.atst_frame.atst_encoder import (
            Model as ATSTModel,
        )

        # ATST requires model path
        # atst_model_path = getattr(
        #     model_config, "atst_model_path", "pretrained/atst_as2M.pt"
        # )
        # atst_model_path = "pretrained/atst_frame_base.pt"
        atst_model_path = "pretrained/atst_as2M.pt"
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
    elif model_name == "eat":
        from representation_learning.models.eat.audio_model import (
            Model as EATModel,  # Local import to avoid heavy deps when unused
        )

        # Optional EAT-specific kwargs
        embed_dim = getattr(model_config, "embed_dim", 768)
        patch_size = getattr(model_config, "patch_size", 16)
        target_length = getattr(model_config, "target_length", 256)
        enable_ema = getattr(model_config, "enable_ema", False)
        pretraining_mode = getattr(model_config, "pretraining_mode", False)
        handle_padding = getattr(model_config, "handle_padding", False)
        eat_cfg_overrides = getattr(model_config, "eat_cfg", None)

        return EATModel(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            embed_dim=embed_dim,
            patch_size=patch_size,
            target_length=target_length,
            enable_ema=enable_ema,
            pretraining_mode=pretraining_mode,
            handle_padding=handle_padding,
            eat_cfg=eat_cfg_overrides,
        )

    elif model_name == "beats":
        from representation_learning.models.beats_model import (
            Model as BeatsModel,
        )

        use_naturelm = getattr(model_config, "use_naturelm", False)

        return BeatsModel(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            use_naturelm=use_naturelm,
        )
    elif model_name == "eat_hf":
        from representation_learning.models.eat_hf import (
            Model as EATHFModel,  # Local import to avoid HF deps when unused
        )

        target_length = getattr(model_config, "target_length", 1024)
        pooling = getattr(model_config, "pooling", "cls")
        model_id = getattr(
            model_config,
            "model_id",
            # "worstchan/EAT-base_epoch30_pretrain",
            "worstchan/EAT-base_epoch30_finetune_AS2M",
        )
        return_features_only = getattr(model_config, "return_features_only", True)

        return EATHFModel(
            model_name=model_id,
            num_classes=num_classes,
            device=model_config.device,
            audio_config=model_config.audio_config,
            target_length=target_length,
            pooling=pooling,
            return_features_only=return_features_only,
        )
    elif model_name == "dummy_model":
        embedding_dim = getattr(model_config, "embedding_dim", 768)
        return_features_only = getattr(model_config, "return_features_only", False)

        return DummyModel(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            device=model_config.device,
            audio_config=model_config.audio_config,
            embedding_dim=embedding_dim,
            return_features_only=return_features_only,
        )
    else:
        # Fallback
        supported = (
            "'efficientnet', 'clip', 'perch', 'perch_bacpipe', 'atst', 'eat', 'eat_hf', 'resnet18', "
            "'resnet50', 'resnet152', 'beats', 'dummy_model'"
        )
        raise NotImplementedError(
            f"Model '{model_name}' is not implemented. Supported models: {supported}"
        )
