import logging
from typing import List, Optional, Union

import torch
from torchaudio.models import wav2vec2_model

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


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
        # Feed-forward layers will be discovered in _discover_linear_layers override

    def _discover_linear_layers(self) -> None:
        """Discover and cache all linear layer names including feed-forward layers.

        This overrides the base class method to discover AVES-specific layers
        beyond just nn.Linear layers.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            # Discover standard linear layers
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self._layer_names.append(name)

            # Discover additional AVES-specific layers (feed-forward layers from
            # transformer blocks)
            # These are typically named like "encoder.layers.0.intermediate_dense",
            # "encoder.layers.0.output_dense"
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Linear) and (
                    "intermediate_dense" in name or "output_dense" in name
                ):
                    if name not in self._layer_names:
                        self._layer_names.append(name)

            logger.debug(
                f"Discovered {len(self._layer_names)} hookable layers in AVES model: "
                f"{self._layer_names}"
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
        *,
        padding_mask: torch.Tensor | None = None,  # noqa: ANN401
        aggregation: str = "none",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from all registered hooks in the AVES model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
            padding_mask: Optional padding mask
            aggregation: Aggregation method for multiple layers ('mean', 'max',
                'cls_token', 'none')

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Model embeddings (tensor if
                aggregation!="none", list if False)

        Raises:
            ValueError: If no hooks are registered or no outputs are captured
        """
        # Check if hooks are registered
        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Hooks are already registered in __init__ via base class

        try:
            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
            else:
                wav = x
                mask = padding_mask

            # Forward pass to trigger hooks
            with torch.no_grad():
                self.forward(wav, mask)

            logger.debug(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []

            for layer_name in self._hook_outputs.keys():
                embeddings.append(self._hook_outputs[layer_name])
                logger.debug(
                    f"Found embedding for {layer_name}: "
                    f"{self._hook_outputs[layer_name].shape}"
                )

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(
                    f"No layers found matching: {self._hook_outputs.keys()}"
                )

            # Process embeddings based on average_over_time parameter
            if aggregation == "none":
                return embeddings
            else:
                # Determine expected batch size from input
                if isinstance(x, dict):
                    expected_batch_size = x["raw_wav"].shape[0]
                else:
                    expected_batch_size = x.shape[0]

                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        # Already in correct shape
                        pass
                    elif embeddings[i].dim() == 3:
                        # Check if tensor is in time-first format
                        # (time, batch, features)
                        if embeddings[i].shape[0] != expected_batch_size:
                            # Transpose to batch-first format
                            embeddings[i] = embeddings[i].view(
                                embeddings[i].shape[0], -1
                            )
                        if aggregation == "mean":
                            embeddings[i] = embeddings[i].mean(dim=1)
                        elif aggregation == "max":
                            embeddings[i] = embeddings[i].max(dim=1)[
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
