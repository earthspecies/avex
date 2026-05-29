"""PrototypicalWrapper: frozen backbone + prototype head for phase-2 training."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from avex.models.probes.prototypical_probe import PrototypicalHead

logger = logging.getLogger(__name__)


class PrototypicalWrapper(nn.Module):
    """Frozen backbone + PrototypicalHead as a single trainable model.

    Used exclusively during phase-2 of the AudioProtoPNet two-stage training
    procedure.  The backbone is permanently frozen; only the prototype head
    is trained.

    Parameters
    ----------
    backbone : nn.Module
        Pretrained backbone with ``extract_embeddings`` and
        ``register_hooks_for_layers`` methods.
    head : PrototypicalHead
        Prototype classification head.
    embedding_layer : str
        Layer name passed to ``register_hooks_for_layers`` for embedding
        extraction. ``"last_layer"`` is a true virtual hook (pre-classifier
        prototype activations) only on AudioProtoPNet / AudioProtoPNetSED.
        On plain ConvNeXt / EfficientNet there is no virtual ``last_layer``;
        it falls back to the base ``_get_last_non_classification_layer``
        discovery, which resolves to the last discovered conv/linear layer.
        Pass an explicit layer name for those backbones if that fallback does
        not select the intended tap.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: PrototypicalHead,
        embedding_layer: str = "last_layer",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.backbone.register_hooks_for_layers([embedding_layer])
        logger.info("PrototypicalWrapper: backbone frozen, hook on '%s'", embedding_layer)

        self.num_classes: int = head.num_classes
        self.device: str = getattr(backbone, "device", "cpu")

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.backbone.extract_embeddings(x, aggregation="mean", freeze_backbone=True)
        return self.head(embeddings)

    def train(self, mode: bool = True) -> "PrototypicalWrapper":
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _infer_dummy_length(backbone: nn.Module, default: int = 32_000 * 5) -> int:
        """Infer a valid dummy-clip length (in samples) from the backbone audio config.

        Returns
        -------
        int
            Number of samples for the dim-probing dummy input. Falls back to
            ``default`` (5 s @ 32 kHz) when no audio config is available.
        """
        ap = getattr(backbone, "audio_processor", None)
        if ap is not None:
            seconds = getattr(ap, "target_length_seconds", None)
            sr = getattr(ap, "sr", None)
            if seconds and sr:
                return int(seconds * sr)
        return default

    @classmethod
    def build(
        cls,
        backbone: nn.Module,
        num_classes: int,
        num_prototypes_per_class: int,
        embedding_layer: str = "last_layer",
    ) -> "PrototypicalWrapper":
        """Probe backbone to get embedding dim, then construct wrapper.

        Returns
        -------
        PrototypicalWrapper
            Wrapper with frozen backbone and freshly initialised prototype head.
        """
        backbone_device = getattr(backbone, "device", "cpu")
        backbone.eval()
        backbone.register_hooks_for_layers([embedding_layer])

        # Prototype-pooled extraction is input-length-independent, so this only
        # needs a valid forward length. Derive it from the backbone's audio
        # config when available instead of assuming 32 kHz x 5 s.
        dummy = torch.zeros(1, cls._infer_dummy_length(backbone), device=backbone_device)
        with torch.no_grad():
            emb = backbone.extract_embeddings(dummy, aggregation="mean", freeze_backbone=True)
        embedding_dim = emb.shape[-1]
        logger.info("Probed embedding_dim=%d from layer '%s'", embedding_dim, embedding_layer)

        head = PrototypicalHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_prototypes_per_class=num_prototypes_per_class,
        )
        return cls(backbone, head, embedding_layer=embedding_layer)
