import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.beats.beats import BEATs, BEATsConfig
from representation_learning.utils import universal_torch_load

logger = logging.getLogger(__name__)

BEATS_PRETRAINED_PATH_FT = (
    "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
BEATS_PRETRAINED_PATH_SSL = (
    "gs://representation-learning/pretrained/BEATs_iter3_plus_AS2M.pt"
)

BEATS_PRETRAINED_PATH_NATURELM = "gs://foundation-models/beats_ckpts/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2_rl_loaded.pt"


class Model(ModelBase):
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

    The underlying BEATs implementation operates directly on raw‐waveform
    inputs.  We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    Notes
    -----
    1.  BEATs extracts a sequence of frame-level embeddings with dimension
        ``cfg.encoder_embed_dim`` (default: ``768``).  We convert this
        variable-length sequence into a fixed-dimensional vector via masked
        mean-pooling before feeding it to a linear classifier.
    2.  When ``return_features_only=True`` the classifier layer is skipped and
        the pooled embedding is returned directly, which is handy for
        representation extraction / linear probing.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        use_naturelm: bool = False,
        fine_tuned: bool = False,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------
        # 1.  Build the BEATs backbone
        # ------------------------------------------------------------------

        if fine_tuned:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_FT
        else:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_SSL

        beats_ckpt = universal_torch_load(
            beats_checkpoint_path, cache_mode="use", map_location="cpu"
        )
        self.use_naturelm = use_naturelm
        self.fine_tuned = fine_tuned
        beats_cfg = BEATsConfig(beats_ckpt["cfg"])
        print(beats_cfg)
        if use_naturelm:  # BEATs-NatureLM has no config, load from regular ckpt first.
            beats_ckpt_naturelm = universal_torch_load(
                BEATS_PRETRAINED_PATH_NATURELM, map_location="cpu"
            )
        else:
            beats_ckpt_naturelm = beats_ckpt["model"]
        # beats_ckpt_naturelm = beats_ckpt
        self.backbone = BEATs(beats_cfg)
        self.backbone.to(device)
        self.backbone.load_state_dict(beats_ckpt_naturelm, strict=False)

        # ------------------------------------------------------------------
        # 2.  Optional classifier for supervised training
        # ------------------------------------------------------------------
        self._return_features_only = return_features_only
        if not return_features_only:
            self.classifier = nn.Linear(768, num_classes)
        else:
            self.register_module("classifier", None)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # 3.  Pre-discover MLP (fc1, fc2) layers for efficient hook management
        # ------------------------------------------------------------------
        # MLP layers will be discovered in _discover_linear_layers override

    def _discover_linear_layers(self) -> None:
        """
        Discover and cache only the BEATs layers that are useful for embeddings.
        This method is called when target_layers=["all"] is used.
        Specifically:
        - backbone.post_extract_proj
        - backbone.encoder.layers.{i}.fc2 (only fc2 layers from encoder blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial projection after conv frontend
                if name.endswith("post_extract_proj"):
                    self._layer_names.append(name)

                # Keep only the fc2 layers from transformer encoder blocks
                # Pattern: backbone.encoder.layers.{i}.fc2
                elif name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    self._layer_names.append(name)

            logger.info(
                f"Discovered {len(self._layer_names)} embedding layers in BEATs: "
                f"{self._layer_names}"
            )

    def _discover_embedding_layers(self) -> None:
        """
        Discover and cache only the BEATs layers that are useful for embeddings.
        Specifically:
        - backbone.post_extract_proj
        - backbone.encoder.layers.{i}.fc2 (only fc2 layers from encoder blocks)
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial projection after conv frontend
                if name.endswith("post_extract_proj"):
                    self._layer_names.append(name)

                # Keep only the fc2 layers from transformer encoder blocks
                # Pattern: backbone.encoder.layers.{i}.fc2
                elif name.endswith(".fc2") and "backbone.encoder.layers." in name:
                    self._layer_names.append(name)

            logger.info(
                f"Discovered {len(self._layer_names)} embedding layers in BEATs: "
                f"{self._layer_names}"
            )

    # ----------------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform with shape ``(batch, time)``.
        padding_mask : torch.Tensor, optional
            Boolean mask where *True* denotes padding elements.  Shape must be
            ``(batch, time)`` and match *x*.

        Returns
        -------
        torch.Tensor
            • When *return_features_only* is **False**: logits of shape
              ``(batch, num_classes)``
            • Otherwise: pooled embeddings of shape
              ``(batch, encoder_embed_dim)``
        """
        # Optional audio pre-processing
        x = self.process_audio(x)

        features, frame_padding = self.backbone(x, padding_mask)

        # features: (B, T', D)
        # frame_padding: (B, T') or None

        # ------------------------------------------------------------------
        # 3.  Masked mean-pooling over the temporal dimension
        # ------------------------------------------------------------------
        if frame_padding is not None and frame_padding.any():
            masked_features = features.clone()
            masked_features[frame_padding] = 0.0  # Zero-out padded frames
            valid_counts = (~frame_padding).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_features.sum(dim=1) / valid_counts
        else:
            pooled = features.mean(dim=1)

        if self._return_features_only:
            return pooled
        else:
            return self.classifier(pooled)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from all registered hooks in the BEATs model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for BEATs)
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none')
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Model embeddings (tensor if aggregation!="none", list if False)

        Raises
        ------
        ValueError
            If input tensor is None or audio tensor is empty
        """
        # Validate input
        if x is None:
            raise ValueError("Input tensor cannot be None")

        # Check for empty audio
        if isinstance(x, dict):
            wav = x["raw_wav"]
        else:
            wav = x

        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise ValueError("Audio tensor cannot be empty")

        # Check if hooks are registered
        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        # Store original training state and set to eval for deterministic results
        was_training = self.training
        self.eval()

        # Set deterministic behavior for CUDA if available
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            # Clear previous hook outputs
            self._clear_hook_outputs()

            # Hooks are already registered in __init__ via base class

            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
            else:
                wav = x
                mask = padding_mask

            # Forward pass to trigger hooks (conditionally use torch.no_grad based on
            # freeze_backbone)
            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, mask)
            else:
                self.forward(wav, mask)

            logger.debug(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            for layer_name in self._hook_outputs.keys():
                embedding = self._hook_outputs[layer_name]

                # Handle different tensor dimensions for sequence probes
                if embedding.dim() == 4:
                    # 4D tensor: reshape to [batch_size, time, dim1*dim2]
                    # Example: [1, 12, 248, 8] -> [1, 248, 96]
                    batch_size, dim1, time_dim, dim2 = embedding.shape
                    embedding = embedding.transpose(1, 2)  # [batch, time, dim1, dim2]
                    embedding = embedding.reshape(
                        batch_size, time_dim, dim1 * dim2
                    )  # [batch_size, time, dim1*dim2]
                    logger.debug(
                        f"Reshaped 4D embedding for {layer_name} from "
                        f"{self._hook_outputs[layer_name].shape} to {embedding.shape}"
                    )
                if embedding.dim() == 3:
                    # 3D tensor: check if it needs transposing
                    original_shape = embedding.shape

                    # Case 1: transpose seq_len and batch_size dimensions
                    if (
                        embedding.shape[1] == wav.shape[0]
                        and embedding.shape[0] != wav.shape[0]
                    ):
                        embedding = embedding.transpose(0, 1)
                        logger.debug(
                            f"Transposed 3D embedding for {layer_name} from "
                            f"{original_shape} to {embedding.shape}"
                        )
                    # Case 2: reshape attention weights [heads, seq_len, seq_len]
                    # This handles attention weight matrices
                    elif (
                        embedding.shape[0] != wav.shape[0]
                        and embedding.shape[1] == embedding.shape[2]
                    ):
                        # This looks like attention weights [heads, seq_len, seq_len]
                        # We need to reshape to [batch_size, seq_len, heads*seq_len]
                        heads, seq_len, _ = embedding.shape
                        embedding = embedding.reshape(
                            1, seq_len, heads * seq_len
                        )  # [1, seq_len, heads*seq_len]
                        logger.debug(
                            f"Reshaped attention weights for {layer_name} from "
                            f"{original_shape} to {embedding.shape}"
                        )
                    # Case 3: reshape attention weights [seq_len, seq_len, heads]
                    elif (
                        embedding.shape[0] == embedding.shape[1]
                        and embedding.shape[2] != wav.shape[0]
                    ):
                        # This looks like attention weights [seq_len, seq_len, heads]
                        seq_len, _, heads = embedding.shape
                        embedding = embedding.reshape(
                            1, seq_len, seq_len * heads
                        )  # [1, seq_len, seq_len*heads]
                        logger.debug(
                            f"Reshaped attention weights for {layer_name} from "
                            f"{original_shape} to {embedding.shape}"
                        )

                embeddings.append(embedding)
                logger.debug(f"Found embedding for {layer_name}: {embedding.shape}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(
                    f"No layers found matching: {self._hook_outputs.keys()}"
                )

            # Process embeddings based on average_over_time parameter
            if aggregation == "none":
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return embeddings
            else:
                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        # Already in correct shape
                        pass
                    elif embeddings[i].dim() == 3:
                        if aggregation == "mean":
                            embeddings[i] = torch.mean(embeddings[i], dim=1)
                        elif aggregation == "max":
                            embeddings[i] = torch.max(embeddings[i], dim=1)[
                                0
                            ]  # max returns (values, indices)
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(
                                f"Unsupported aggregation method: {aggregation}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {embeddings[i].dim()}. "
                            f"Expected 2 or 3."
                        )

                # Concatenate all embeddings
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return torch.cat(embeddings, dim=1)

        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()
            # Restore original training state
            if was_training:
                self.train()

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        audio = super().process_audio(x)
        if self.use_naturelm:
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
