from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.beats.beats import BEATs, BEATsConfig
from representation_learning.utils import universal_torch_load

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
        self._mlp_layer_names: List[str] = []
        self._linear_layer_names: List[str] = []  # For backward compatibility
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._linear_layer_names.append(name)
                if name.endswith("fc1") or name.endswith("fc2"):
                    self._mlp_layer_names.append(name)
        print(
            f"Discovered {len(self._mlp_layer_names)} MLP (fc1/fc2) layers "
            f"for hook management"
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
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Extract embeddings from the model with automatic batch splitting.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        layers : List[str]
            List of layer names. If 'all' is included, all MLP (fc1/fc2) layers in the
            model will be automatically found and used.
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for BEATs)
        average_over_time : bool
            Whether to average embeddings over time dimension
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none')

        Returns
        -------
        torch.Tensor
            Concatenated embeddings from the requested layers

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

            # Handle empty layers list - return main features
            if not layers:
                if isinstance(x, dict):
                    wav = x["raw_wav"]
                    mask = x.get("padding_mask")
                else:
                    wav = x
                    mask = padding_mask

                with torch.no_grad():
                    return self.forward(wav, mask)

            # Handle 'all' case - use pre-discovered MLP layers
            target_layers = layers.copy()
            if "all" in layers:
                print(
                    "'all' specified in layers, using pre-discovered MLP (fc1/fc2) "
                    "layers for BEATs model..."
                )

                if self._mlp_layer_names:
                    print(
                        f"Using {len(self._mlp_layer_names)} pre-discovered MLP layers"
                    )
                    # Get specific layers (excluding 'all')
                    specific_layers = [layer for layer in layers if layer != "all"]
                    # Combine specific layers with MLP layers, avoiding duplicates
                    target_layers = list(
                        dict.fromkeys(specific_layers + self._mlp_layer_names)
                    )
                else:
                    print("No MLP layers found in the model")

                print(
                    f"Target layers after 'all' expansion: {len(target_layers)} layers"
                )

            # Register hooks for requested layers
            self._register_hooks_for_layers(target_layers)

            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
            else:
                wav = x
                mask = padding_mask

            print(f"Starting forward pass with target layers: {target_layers}")

            # Forward pass to trigger hooks
            with torch.no_grad():
                self.forward(wav, mask)

            print(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            print(f"Collecting embeddings from {len(target_layers)} target layers")
            for layer_name in target_layers:
                if layer_name in self._hook_outputs:
                    embeddings.append(self._hook_outputs[layer_name])
                    print(
                        f"Found embedding for {layer_name}: "
                        f"{self._hook_outputs[layer_name].shape}"
                    )
                else:
                    print(f"No output captured for layer: {layer_name}")

            print(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {target_layers}")

            if average_over_time:
                result = []
                # Determine expected batch size from input
                if isinstance(x, dict):
                    expected_batch_size = x["raw_wav"].shape[0]
                else:
                    expected_batch_size = x.shape[0]

                for emb in embeddings:
                    if emb.dim() == 2:
                        # Already in correct shape, just append
                        result.append(emb)
                    elif emb.dim() == 3:
                        # Check if tensor is in time-first format
                        # (time, batch, features)
                        if emb.shape[0] != expected_batch_size:
                            # Transpose to batch-first format
                            emb = emb.transpose(0, 1)

                        # Apply aggregation method
                        if aggregation == "mean":
                            aggregated = torch.mean(emb, dim=1)
                        elif aggregation == "max":
                            aggregated = torch.max(emb, dim=1)[0]
                        elif aggregation == "cls_token":
                            # For transformer models, take first token
                            aggregated = emb[:, 0, :]
                        elif aggregation == "none":
                            # Keep full sequence
                            aggregated = emb
                        else:
                            raise ValueError(
                                f"Unknown aggregation method: {aggregation}"
                            )
                        result.append(aggregated)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )

                # Apply aggregation across layers
                if len(result) > 1:
                    if aggregation == "mean":
                        return torch.cat(result, dim=1)
                    elif aggregation == "max":
                        return torch.cat(result, dim=1)
                    elif aggregation == "cls_token":
                        return torch.cat(result, dim=1)
                    elif aggregation == "none":
                        # For 'none', we need to handle multiple layers differently
                        # Try to stack, but if sizes don't match, fall back to
                        # concatenation
                        try:
                            return torch.stack(result, dim=1)
                        except RuntimeError:
                            # If stacking fails due to different sizes, flatten and
                            # concatenate instead
                            flattened_results = []
                            for emb in result:
                                if emb.dim() == 3:
                                    # Flatten the 3D tensor to 2D
                                    batch_size, seq_len, features = emb.shape
                                    flattened = emb.reshape(batch_size, -1)
                                    flattened_results.append(flattened)
                                else:
                                    flattened_results.append(emb)
                            return torch.cat(flattened_results, dim=1)
                else:
                    return result[0]
            else:
                # For non-averaged case, also transpose time-first tensors
                result = []
                # Determine expected batch size from input
                if isinstance(x, dict):
                    expected_batch_size = x["raw_wav"].shape[0]
                else:
                    expected_batch_size = x.shape[0]

                for emb in embeddings:
                    if emb.dim() == 2:
                        result.append(emb)
                    elif emb.dim() == 3:
                        # Check if tensor is in time-first format
                        if emb.shape[0] != expected_batch_size:
                            # Transpose to batch-first format
                            emb = emb.transpose(0, 1)
                        result.append(emb)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )

                # Apply aggregation across layers
                if len(result) > 1 and aggregation == "none":
                    # Stack along a new dimension to preserve layer information
                    return torch.stack(result, dim=1)
                else:
                    return result

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
