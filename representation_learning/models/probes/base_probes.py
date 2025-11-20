"""Base classes for  probes (2D and 3D variants)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


class _BaseProbe(ABC, nn.Module):
    """Common scaffolding for probes."""

    def __init__(
        self,
        *,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[Union[int, Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        aggregation: str = "mean",
        target_length: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.num_classes = num_classes
        self.feature_mode = feature_mode
        self.aggregation = aggregation
        self.target_length = target_length
        self.freeze_backbone = freeze_backbone

        # Note: Do not predefine optional attributes; create them only if needed

        inferred_dim, num_embeddings = self._setup_projections_and_infer_dim(input_dim=input_dim)

        self.build_head(inferred_dim)
        self.to(device)
        logger.info(
            "Probe initialized: layers=%s, feature_mode=%s, aggregation=%s, "
            "freeze_backbone=%s, target_length=%s, inferred_dim=%s, "
            "num_classes=%s, num_embeddings=%s, has_layer_weights=%s",
            self.layers,
            self.feature_mode,
            self.aggregation,
            self.freeze_backbone,
            self.target_length,
            inferred_dim,
            num_classes,
            num_embeddings,
            hasattr(self, "layer_weights"),
        )

    @abstractmethod
    def _expected_dimensionality(self) -> int:
        """Return target embedding rank (2 or 3)."""

    @abstractmethod
    def build_head(self, inferred_dim: int) -> None:
        """Create architecture-specific head given inferred input dim."""

    @abstractmethod
    def forward(
        self, x: Union[torch.Tensor, dict], padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Subclasses must provide the end-to-end forward pass."""

    def _get_dummy_embeddings_from_input_dim(
        self, input_dim: Union[int, Tuple[int, ...], List[Tuple[int, ...]]]
    ) -> List[torch.Tensor]:
        if isinstance(input_dim, list):
            return [torch.randn(1, *shape, device=self.device) for shape in input_dim]
        if isinstance(input_dim, tuple):
            return [torch.randn(1, *input_dim, device=self.device)]
        if self._expected_dimensionality() == 3:
            # For 3D expectations, treat the provided int as feature dim
            # and create a minimal sequence length of 1:
            # (batch, seq=1, features=input_dim)
            return [torch.randn(1, 1, input_dim, device=self.device)]
        return [torch.randn(1, input_dim, device=self.device)]

    def _infer_target_length(self) -> int:
        if self.target_length is not None:
            return int(self.target_length)
        assert self.base_model is not None
        audio_processor = self.base_model.audio_processor
        if hasattr(audio_processor, "target_length_seconds"):
            return int(audio_processor.target_length_seconds * audio_processor.sr)
        if hasattr(audio_processor, "target_length"):
            return int(audio_processor.target_length)
        raise ValueError(
            "target_length must be provided when base_model.audio_processor "
            "does not have target_length or target_length_seconds"
        )

    def _extract_dummy_embeddings_from_model(self) -> TensorOrList:
        assert self.base_model is not None
        with torch.no_grad():
            dummy = torch.randn(1, self._infer_target_length(), device=self.device)
            return self.base_model.extract_embeddings(dummy, aggregation=self.aggregation)

    def _setup_projections_and_infer_dim(
        self,
        *,
        input_dim: Optional[Union[int, Tuple[int, ...], List[Tuple[int, ...]]]],
    ) -> Tuple[int, int]:
        inferred_dim: Optional[int] = None
        num_embeddings: int = 1

        if self.feature_mode and input_dim is not None:
            dummy_embeddings: TensorOrList = self._get_dummy_embeddings_from_input_dim(input_dim)
            if len(dummy_embeddings) == 1:
                dummy_embeddings = dummy_embeddings[0]
        else:
            if self.feature_mode and self.base_model is None and input_dim is None:
                raise ValueError(
                    "input_dim must be provided when feature_mode=True and base_model is None"
                )
            dummy_embeddings = self._extract_dummy_embeddings_from_model()

        if isinstance(dummy_embeddings, list):
            dummy_embeddings = [emb.detach() for emb in dummy_embeddings]
            num_embeddings = len(dummy_embeddings)
            inferred_dim = self._analyze_and_create_projectors(dummy_embeddings)
            # Only create layer weights when there are multiple embeddings
            if num_embeddings > 1:
                self.layer_weights = nn.Parameter(torch.zeros(num_embeddings))
        else:
            if self.freeze_backbone:
                dummy_embeddings = dummy_embeddings.detach()
            inferred_dim = self._infer_single_tensor_dim(dummy_embeddings)

        assert inferred_dim is not None
        return inferred_dim, num_embeddings

    @abstractmethod
    def _analyze_and_create_projectors(self, embeddings: List[torch.Tensor]) -> int:
        """Inspect list embeddings, create projectors if needed, return feature dim."""

    @abstractmethod
    def _infer_single_tensor_dim(self, emb: torch.Tensor) -> int:
        """Infer feature dimension for single tensor embeddings."""

    def _get_embeddings(
        self, x: Union[torch.Tensor, dict], padding_mask: Optional[torch.Tensor]
    ) -> TensorOrList:
        if self.feature_mode:
            if isinstance(x, dict):
                # Prefer explicit raw_wav if present for feature mode dicts
                if "raw_wav" in x:
                    return x["raw_wav"]
                # Exclude non-embedding keys commonly present
                embed_keys = [k for k in x.keys() if k not in ("label", "padding_mask")]
                if len(embed_keys) == 1:
                    return x[embed_keys[0]]
                return [x[k] for k in embed_keys]
            return x  # type: ignore[return-value]

        if isinstance(x, dict):
            padding_mask = x.get("padding_mask")
            x = x["raw_wav"]
        assert self.base_model is not None
        embeddings = self.base_model.extract_embeddings(
            x,
            padding_mask=padding_mask,
            aggregation=self.aggregation,
            freeze_backbone=self.freeze_backbone,
        )
        if self.freeze_backbone:
            if isinstance(embeddings, list):
                embeddings = [emb.detach() for emb in embeddings]
            else:
                embeddings = embeddings.detach()
        return embeddings

    def _sum(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        # If no learned weights, default to uniform average
        if not hasattr(self, "layer_weights") or self.layer_weights is None:
            weights = torch.ones(len(embeddings), device=embeddings[0].device)
        else:
            weights = torch.softmax(self.layer_weights, dim=0)
        out = torch.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights, strict=False):
            out = out + w * emb
        return out

    def get_learned_weights_table(self) -> str:
        """Get learned combination weights as a formatted table string.

        Returns a concise message when no weights exist, otherwise returns a
        formatted table with raw and normalized weights per layer.

        Returns
        -------
        str
            Formatted weights table or message about no weights
        """
        if not hasattr(self, "layer_weights") or getattr(self, "layer_weights", None) is None:
            return "No learned weights found. This probe does not use weighted sum of embeddings."

        raw = self.layer_weights.detach().cpu().numpy()
        norm = torch.softmax(self.layer_weights, dim=0).detach().cpu().numpy()

        # Build the weights table as a single string for atomic logging
        weights_table = []
        weights_table.append("Learned Layer Weights:")
        weights_table.append("=" * 50)
        weights_table.append(
            f"{'Layer':<15} {'Raw Weight':<12} {'Normalized':<12} {'Percentage':<12}"
        )
        weights_table.append("-" * 50)
        for i, (r, n) in enumerate(zip(raw, norm, strict=False)):
            # Always use generic labels to avoid leaking config placeholders
            # like 'all' or mismatched lengths between weights and self.layers
            layer_name = f"Layer_{i}"
            pct = n * 100
            weights_table.append(f"{layer_name:<15} {r:<12.4f} {n:<12.4f} {pct:<12.2f}%")
        weights_table.append("-" * 50)
        weights_table.append("Sum of normalized weights: %.6f" % norm.sum())
        weights_table.append("Number of layers: %d" % len(raw))

        return "\n".join(weights_table)

    @abstractmethod
    def _combine_or_reshape_embeddings(self, embeddings: TensorOrList) -> torch.Tensor:
        """Prepare embeddings for the head: combine layers and normalize shape."""


class BaseProbe2D(_BaseProbe):
    """Base for 2D probes that expect (batch, features) inputs."""

    def _expected_dimensionality(self) -> int:
        return 2

    def _analyze_and_create_projectors(self, embeddings: List[torch.Tensor]) -> int:
        if not embeddings:
            raise ValueError(
                "No embeddings provided to _analyze_and_create_projectors. "
                "This usually indicates that no layers were found in the base model. "
                "Please check that the target_layers are valid for the model "
                "architecture."
            )

        flattened_dims: List[int] = []
        for emb in embeddings:
            if emb.dim() == 4:
                flattened_dims.append(emb.shape[1] * emb.shape[2] * emb.shape[3])
            elif emb.dim() == 3:
                flattened_dims.append(emb.shape[1] * emb.shape[2])
            elif emb.dim() == 2:
                flattened_dims.append(emb.shape[1])
            else:
                raise ValueError(f"Unsupported embedding dim {emb.dim()} for 2D probe")

        from collections import Counter

        counts = Counter(flattened_dims)
        most_common_dim, count = counts.most_common(1)[0]
        if count > len(flattened_dims) / 2:
            target_dim = most_common_dim
        else:
            target_dim = max(flattened_dims)

        self.embedding_projectors = nn.ModuleList()
        for in_dim in flattened_dims:
            if in_dim == target_dim:
                self.embedding_projectors.append(None)  # type: ignore[arg-type]
            else:
                self.embedding_projectors.append(nn.Linear(in_dim, target_dim))
        return target_dim

    def _infer_single_tensor_dim(self, emb: torch.Tensor) -> int:
        if emb.dim() == 2:
            return emb.shape[-1]
        if emb.dim() == 3:
            return emb.shape[1] * emb.shape[2]
        if emb.dim() == 4:
            return emb.shape[1] * emb.shape[2] * emb.shape[3]
        raise ValueError(f"Linear probe expects 2D, 3D or 4D embeddings, got {emb.shape}")

    def _combine_or_reshape_embeddings(self, embeddings: TensorOrList) -> torch.Tensor:
        if isinstance(embeddings, list):
            projected: List[torch.Tensor] = []
            assert self.embedding_projectors is not None
            for emb, projector in zip(embeddings, self.embedding_projectors, strict=False):
                if emb.dim() == 4:
                    emb2 = emb.reshape(emb.shape[0], -1)
                elif emb.dim() == 3:
                    emb2 = emb.reshape(emb.shape[0], -1)
                elif emb.dim() == 2:
                    emb2 = emb
                else:
                    raise ValueError(f"Unsupported embedding dim {emb.dim()} for 2D probe")
                emb2 = projector(emb2) if projector is not None else emb2
                projected.append(emb2)
            return self._sum(projected)

        if embeddings.dim() == 3:
            return embeddings.reshape(embeddings.shape[0], -1)
        if embeddings.dim() == 4:
            return embeddings.reshape(embeddings.shape[0], -1)
        if embeddings.dim() == 2:
            return embeddings
        raise ValueError(f"Linear probe expects 2D, 3D or 4D embeddings, got {embeddings.shape}")


class BaseProbe3D(_BaseProbe):
    """Base for 3D probes that expect (batch, seq_len, features) inputs."""

    def _expected_dimensionality(self) -> int:
        return 3

    def _analyze_and_create_projectors(self, embeddings: List[torch.Tensor]) -> int:
        if not embeddings:
            raise ValueError(
                "No embeddings provided to _analyze_and_create_projectors. "
                "This usually indicates that no layers were found in the base model. "
                "Please check that the target_layers are valid for the model "
                "architecture."
            )

        info: List[Tuple[int, int, int]] = []  # (rank, seq_len, feat)
        for emb in embeddings:
            if emb.dim() == 3:
                info.append((3, emb.shape[1], emb.shape[2]))
            elif emb.dim() == 4:
                info.append((4, emb.shape[3], emb.shape[1] * emb.shape[2]))
            elif emb.dim() == 2:
                info.append((2, emb.shape[1], 1))
            else:
                raise ValueError(f"Unsupported embedding dim {emb.dim()} for 3D probe")

        from collections import Counter

        seq_counts = Counter([s for _, s, _ in info])
        feat_counts = Counter([f for _, _, f in info])
        most_seq, seq_count = seq_counts.most_common(1)[0]
        most_feat, feat_count = feat_counts.most_common(1)[0]
        target_seq = most_seq if seq_count > len(info) / 2 else max(s for _, s, _ in info)
        target_feat = most_feat if feat_count > len(info) / 2 else max(f for _, _, f in info)

        projectors: List[Optional[nn.Linear]] = []
        for _, seq, feat in info:
            if feat == target_feat and seq == target_seq:
                projectors.append(None)
            else:
                projectors.append(nn.Linear(feat, target_feat))
        self.embedding_projectors = nn.ModuleList(projectors)
        return target_feat

    def _format_to_seq_feat(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 3:
            return emb
        if emb.dim() == 4:
            b, c, h, w = emb.shape
            return emb.transpose(1, 3).reshape(b, w, c * h)
        if emb.dim() == 2:
            return emb.unsqueeze(2)
        raise ValueError(f"Unsupported embedding dim {emb.dim()} for 3D probe")

    def _infer_single_tensor_dim(self, emb: torch.Tensor) -> int:
        if emb.dim() == 2:
            return 1
        if emb.dim() == 3:
            return emb.shape[-1]
        if emb.dim() == 4:
            return emb.shape[1] * emb.shape[2]
        raise ValueError(f"Attention probe expects 2D, 3D or 4D embeddings, got {emb.shape}")

    def _combine_or_reshape_embeddings(self, embeddings: TensorOrList) -> torch.Tensor:
        if isinstance(embeddings, list):
            projected: List[torch.Tensor] = []
            assert self.embedding_projectors is not None
            for emb, projector in zip(embeddings, self.embedding_projectors, strict=False):
                emb3 = self._format_to_seq_feat(emb)
                emb3 = projector(emb3) if projector is not None else emb3
                projected.append(emb3)
            # If sequence lengths differ, interpolate to the shortest length
            seq_lens = [p.shape[1] for p in projected]
            target_seq = min(seq_lens)
            if len(set(seq_lens)) > 1:
                aligned: List[torch.Tensor] = []
                for p in projected:
                    if p.shape[1] != target_seq:
                        p = torch.nn.functional.interpolate(
                            p.transpose(1, 2),
                            size=target_seq,
                            mode="linear",
                            align_corners=False,
                        ).transpose(1, 2)
                    aligned.append(p)
                projected = aligned
            return self._sum(projected)

        return self._format_to_seq_feat(embeddings)
