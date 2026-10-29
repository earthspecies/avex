"""BirdMAE model implementation for bioacoustic representation learning.

This module provides the BirdMAE (Bird Masked Autoencoder) model implementation
for bioacoustic representation learning tasks.

The model and feature extractor code is loaded from the Hugging Face Hub with
``trust_remote_code=True``. Known repositories are pinned to a specific commit
(see ``_PINNED_REVISIONS``) so upstream changes cannot silently alter outputs.

The upstream model code was written for transformers 4.x and never calls
``post_init()``, which transformers >= 5.0 requires (loading otherwise fails
with ``AttributeError: ... 'all_tied_weights_keys'``). ``_load_model_class``
wraps the remote class and calls ``post_init()`` when upstream did not.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from avex.configs import AudioConfig
from avex.models.base_model import ModelBase

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "DBD-research-group/Bird-MAE-Base"

# Hub commits of the known Bird-MAE repositories. Bump deliberately after
# checking that embeddings are unchanged (or that the change is wanted).
_PINNED_REVISIONS: dict[str, str] = {
    "DBD-research-group/Bird-MAE-Base": "6cc416d1a7ae2af29b6b866499b3b047a8f01304",
    "DBD-research-group/Bird-MAE-huge": "61106f4d15a61fce956faef80cf17894e1ab9a48",
}


def _load_model_class(model_id: str, revision: Optional[str]) -> tuple[type, Any]:
    """Resolve the remote BirdMAE class and wrap it for transformers >= 5.

    Returns
    -------
    tuple[type, Any]
        The wrapped model class and the model config.

    Raises
    ------
    ValueError
        If ``model_id`` is not a remote-code repository exposing ``AutoModel``.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    class_ref = (getattr(config, "auto_map", None) or {}).get("AutoModel")
    if not class_ref:
        raise ValueError(
            f"BirdMAE model_id '{model_id}' is not a remote-code Bird-MAE repository "
            "(config.auto_map has no 'AutoModel' entry)."
        )
    remote_cls = get_class_from_dynamic_module(class_ref, model_id, revision=revision)

    class _BirdMAEModel(remote_cls):  # type: ignore[valid-type, misc]
        def post_init(self) -> None:
            self._avex_post_init_called = True
            super().post_init()

        def __init__(self, config: Any) -> None:  # noqa: ANN401
            super().__init__(config)
            # No-op once upstream calls post_init() itself.
            if not getattr(self, "_avex_post_init_called", False):
                self.post_init()

    _BirdMAEModel.__name__ = remote_cls.__name__
    _BirdMAEModel.__qualname__ = remote_cls.__qualname__
    return _BirdMAEModel, config


def _fit_time_axis(features: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad (with the batch minimum) or crop the time axis of (B, 1, T, M) features to ``target_length``.

    The pinned Bird-MAE-huge feature extractor compares ``target_length`` against
    the batch dimension instead of the time dimension, so it leaves batches of
    ``>= target_length`` clips unpadded. This mirrors the intended upstream
    behaviour independent of batch size.

    Returns
    -------
    torch.Tensor
        Features with ``features.shape[2] == target_length``.
    """
    length = features.shape[2]
    if length > target_length:
        return features[:, :, :target_length, :]
    if length < target_length:
        return F.pad(features, (0, 0, 0, target_length - length), value=features.min().item())
    return features


class Model(ModelBase):
    """BirdMAE model for bioacoustic representation learning.

    Implements a masked autoencoder model specifically designed for
    bioacoustic representation learning tasks.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        model_id: Optional[str] = None,
    ) -> None:
        """Initialize BirdMAE.

        Parameters
        ----------
        num_classes : Optional[int]
            Number of output classes for the linear head. None or 0 disables it.
        pretrained : bool
            If True, load the Hub weights. If False, build the architecture
            from the Hub config with random weights. Note that ``load_model``
            does not forward this flag (it always uses the default), so a
            ``checkpoint_path`` load downloads the Hub weights before
            overwriting them; ``get_model`` honours it.
        device : str
            Device to run on.
        audio_config : Optional[AudioConfig]
            Audio configuration.
        return_features_only : bool
            If True, return embeddings instead of logits.
        model_id : Optional[str]
            Hugging Face repository ID. Defaults to ``DEFAULT_MODEL_ID``.

        Raises
        ------
        ImportError
            If transformers is not installed.
        """
        super().__init__(device=device, audio_config=audio_config)

        try:
            from transformers import AutoFeatureExtractor
        except ImportError as e:
            raise ImportError("transformers library is required for BirdMAE.") from e

        self.return_features_only = return_features_only
        self.gradient_checkpointing = False
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.revision = _PINNED_REVISIONS.get(self.model_id)
        if self.revision is None:
            logger.warning(f"BirdMAE model_id '{self.model_id}' has no pinned revision; using the Hub default branch.")

        model_cls, config = _load_model_class(self.model_id, self.revision)
        if pretrained:
            self.model = model_cls.from_pretrained(self.model_id, revision=self.revision)
        else:
            self.model = model_cls(config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        )
        self.model = self.model.to(device)
        self._embedding_dim = config.embed_dim

        if not return_features_only and num_classes is not None and num_classes > 0:
            self.num_classes = num_classes
            self.classifier = nn.Linear(self._embedding_dim, num_classes).to(self.device)
        else:
            self.num_classes = None
            self.classifier = None

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio input using BirdMAE's feature extractor.

        The upstream feature extractor pads or truncates to 5 s
        (``feature_extractor.target_length`` frames). Already-processed
        4-D features are returned unchanged, so this is safe to call twice
        (``ModelBase.batch_inference`` does).

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor of shape (batch_size, time_steps), or features
            of shape (batch_size, 1, target_length, n_mels)

        Returns
        -------
        torch.Tensor
            Mel filterbank tensor of shape (batch_size, 1, target_length, n_mels)
        """
        if x.dim() == 4:
            return x.to(self.device)
        features = self.feature_extractor(x.detach().float().cpu(), return_tensors="pt")
        if isinstance(features, dict):
            features = features["input_values"]
        features = _fit_time_axis(features, self.feature_extractor.target_length)
        return features.to(self.device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing over the transformer blocks.

        The upstream model does not implement transformers' gradient
        checkpointing, so each block's forward is wrapped in
        ``torch.utils.checkpoint`` when gradients are enabled.
        """
        self.gradient_checkpointing = True
        for blk in self.model.blocks:
            if getattr(blk, "_avex_checkpointed", False):
                continue
            orig_forward = blk.forward

            def forward(*args: Any, _orig: Any = orig_forward, **kwargs: Any) -> Any:  # noqa: ANN401
                if torch.is_grad_enabled():
                    return torch.utils.checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                return _orig(*args, **kwargs)

            blk.forward = forward
            blk._avex_checkpointed = True

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream returns the globally pooled embedding as last_hidden_state.
        return self.model(self.process_audio(x)).last_hidden_state

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (not used by BirdMAE)

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape (batch_size, embedding_dim), or logits
            when a classifier is attached.
        """
        embeddings = self._embed(x)
        if self.return_features_only or self.classifier is None:
            return embeddings
        return self.classifier(embeddings)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings from BirdMAE.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (not used by BirdMAE)
        aggregation : str
            Aggregation method for compatibility with other backbones. Since BirdMAE
            exposes a single pooled embedding, all supported values are a no-op:
            "none" | "mean" | "max" | "cls_token".
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Model embeddings of shape (batch_size, embedding_dim).

        Raises
        ------
        ValueError
            If an unsupported aggregation method is provided.
        """
        if aggregation not in {"none", "mean", "max", "cls_token"}:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        if isinstance(x, dict):
            x = x["raw_wav"]

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.set_grad_enabled(not freeze_backbone):
                return self._embed(x)
        finally:
            if was_training:
                self.model.train()
