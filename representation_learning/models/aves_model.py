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
    ) -> torch.Tensor:
        if isinstance(x, dict):
            sequence_embeddings = self.forward(x["raw_wav"], padding_mask)
        else:
            sequence_embeddings = self.forward(x, padding_mask)
        return sequence_embeddings.mean(dim=1)  # (B, C)
