from typing import List, Optional

import torch
from torchaudio.models import wav2vec2_model

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


class AVESConfig:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        # Extractor configuration
        self.extractor_mode: str = "group_norm"  # mode for feature extractor
        self.extractor_conv_layer_config: list = [  # configuration for conv layers
            [512, 10, 5],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 2, 2],
            [512, 2, 2],
        ]
        self.extractor_conv_bias: bool = False  # include bias in conv encoder

        # Encoder configuration
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_projection_dropout: float = 0.1  # dropout for encoder projection
        self.encoder_pos_conv_kernel: int = 128  # kernel size for positional conv
        self.encoder_pos_conv_groups: int = 16  # number of groups for positional conv
        self.encoder_num_layers: int = 12  # number of encoder layers
        self.encoder_num_heads: int = 12  # number of attention heads
        self.encoder_attention_dropout: float = 0.1  # dropout for attention
        self.encoder_ff_interm_features: int = 3072  # intermediate features in FFN
        self.encoder_ff_interm_dropout: float = 0.0  # dropout for intermediate FFN
        self.encoder_dropout: float = 0.1  # dropout for encoder
        self.encoder_layer_norm_first: bool = False  # apply layer norm first
        self.encoder_layer_drop: float = 0.05  # probability of dropping encoder layers

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict) -> None:
        self.__dict__.update(cfg)

    def to_dict(self) -> dict:
        return self.__dict__


class Model(ModelBase):
    """Wrapper that adapts the raw *AVES* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

    The underlying AVES implementation operates directly on raw‐waveform
    inputs.  We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    """

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained: bool = False,  # Currently unused; placeholder for future.
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self.config = AVESConfig()

        self.model = wav2vec2_model(**self.config.to_dict(), aux_num_out=None)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/esp-public-files/"
            "birdaves/birdaves-biox-base.torchaudio.pt",
            map_location=device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        # ------------------------------------------------------------------
        # Pre-discover feed-forward (intermediate_dense, output_dense) layers
        # for efficient hook management
        # ------------------------------------------------------------------
        self._mlp_layer_names: List[str] = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                "intermediate_dense" in name or "output_dense" in name
            ):
                self._mlp_layer_names.append(name)
        print(
            f"Discovered {len(self._mlp_layer_names)} feed-forward "
            f"(intermediate_dense/output_dense) layers for hook management"
        )

    def _prep_input(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        return inputs.to(self.device)

    # ----------------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : torch.Tensor
            Padding mask for the input

        Returns
        -------
        torch.Tensor
            Model output (logits or features based on init flag)
        """
        # Optional audio pre-processing
        x = self._prep_input(x)

        features = self.model.extract_features(x)[0]
        return features[-1]

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],  # noqa: ANN401
        layers: List[str],
        *,
        padding_mask: torch.Tensor | None = None,  # noqa: ANN401
        masked_mean: bool = False,
        average_over_time: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from specified layers of the AVES model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
                    layers: List of layer names to extract embeddings from. If 'all' is
                   included, all feed-forward (intermediate_dense and output_dense)
                   layers will be used for comprehensive representation extraction.
            padding_mask: Optional padding mask
            masked_mean: Whether to use masked mean pooling (kept for compatibility)
            average_over_time: Whether to average embeddings over time dimension

        Returns:
            torch.Tensor: Concatenated embeddings from the requested layers

        Raises:
            ValueError: If none of the supplied layers are found in the model
        """
        # Handle empty layers list - return main features
        if not layers:
            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
            else:
                wav = x
                mask = padding_mask

            with torch.no_grad():
                emb = self.forward(wav, mask)
                # Average over time dimension if it's 3D
                if emb.dim() == 3:
                    if average_over_time:
                        emb = emb.mean(dim=1)  # Average over time dimension
                    else:
                        # Return as list for consistency with other cases
                        return [emb]
                else:
                    if not average_over_time:
                        # Return as list for consistency
                        return [emb]
            return emb

        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Handle 'all' case - use all feed-forward (intermediate_dense, output_dense)
        # layers for comprehensive representations
        # If 'all' is not in layers, use the exact layers specified
        target_layers = layers.copy()
        if "all" in layers:
            print(
                "'all' specified in layers, using pre-discovered feed-forward "
                "(intermediate_dense/output_dense) layers for AVES model..."
            )

            if self._mlp_layer_names:
                print(
                    f"Using {len(self._mlp_layer_names)} pre-discovered "
                    f"feed-forward layers"
                )
                target_layers = [
                    layer for layer in layers if layer != "all"
                ] + self._mlp_layer_names
                print(
                    f"Target layers after 'all' expansion: {len(target_layers)} layers"
                )
            else:
                print(
                    "No feed-forward (intermediate_dense/output_dense) layers "
                    "found in AVES model"
                )
                # Fallback to main features when no MLP layers found
                if isinstance(x, dict):
                    wav = x["raw_wav"]
                    mask = x.get("padding_mask")
                else:
                    wav = x
                    mask = padding_mask

                with torch.no_grad():
                    emb = self.forward(wav, mask)
                    # Average over time dimension if it's 3D
                    if emb.dim() == 3:
                        if average_over_time:
                            emb = emb.mean(dim=1)  # Average over time dimension
                        else:
                            # Return as list for consistency with other cases
                            return [emb]
                    else:
                        if not average_over_time:
                            # Return as list for consistency
                            return [emb]
                return emb

        # Register hooks for requested layers (only if not already registered)
        self._register_hooks_for_layers(target_layers)

        try:
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
                        aggregated = torch.mean(emb, dim=1)
                        result.append(aggregated)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )
                return torch.cat(result, dim=1)
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
                return result

        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()
