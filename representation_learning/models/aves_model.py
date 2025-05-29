from typing import Any, List, Optional

import torch
import torch.nn.functional as F
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
    (e.g. ``efficientnetb0.py``) so that it can be selected via
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
        
        # Ensure features is a tensor, not a list
        if isinstance(features, list):
            features = torch.stack(features, dim=0) if len(features) > 1 else features[0]

        return features



    def extract_embeddings(self, x: Any | dict[str, Any], layers: List[str], *, padding_mask: Any | None = None) -> torch.Tensor:
        # ------------------------------------------------------------------ #
        #  Construct / down-sample padding mask (if supplied)               #
        # ------------------------------------------------------------------ #

        # Allow callers to supply a dict with explicit padding ‑mask.
        if padding_mask is None and isinstance(x, dict) and "padding_mask" in x:
            padding_mask = x["padding_mask"]

        frame_mask: Optional[torch.Tensor] = None
        if padding_mask is not None:
            # The raw *padding_mask* is sample-level (shape: B × T_samples) where
            # **True** denotes padded samples.  We need to bring this down to the
            # feature-frame resolution produced by the AVES convolutional front-end.
            # Each conv stride is 320 samples, so we emulate BEATs' approach and
            # use a 1-D max-pool across that window.
            frame_mask = F.max_pool1d(
                padding_mask.float().unsqueeze(1),  # (B, 1, T_samples)
                kernel_size=320,
                stride=320,
            ) > 0  # (B, T_frames)

            # Invert semantics → *True* means **keep** (non-padded) so it aligns
            # with the helper below.
            frame_mask = ~frame_mask

        # ------------------------------------------------------------------ #
        #  Forward pass                                                     #
        # ------------------------------------------------------------------ #

        if isinstance(x, dict):
            sequence_embeddings = self.forward(x["raw_wav"], padding_mask)
        else:
            sequence_embeddings = self.forward(x, padding_mask)

        # ------------------------------------------------------------------ #
        #  Pool over time dimension                                         #
        # ------------------------------------------------------------------ #

        if frame_mask is None:
            return sequence_embeddings.mean(dim=1)  # (B, C)

        return _masked_mean(sequence_embeddings, frame_mask)
 
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a sequence while ignoring padded frames.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(B, T, C)``.
    mask : torch.Tensor
        Boolean tensor of shape ``(B, T)`` where **True** denotes frames to
        *keep* (i.e. non-padded).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, C)`` – mean of the unmasked frames.
    """
    
    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected x to be a torch.Tensor, but got {type(x)}. Value: {x}")
    
    # Add broadcast-friendly channel dimension
    mask_expanded = mask.unsqueeze(-1).type_as(x)  # (B, T, 1)

    # Align *x* with the (down-sampled) mask length if needed.
    expected_len = mask.shape[1]
    if x.shape[1] < expected_len:
        # Rare corner-case when conv-stride rounding leads to an off-by-one
        # mismatch – pad with reflection like upstream.*
        x = F.pad(x, (0, 0, 0, expected_len - x.shape[1]), mode="reflect")
    x = x[:, :expected_len, :]

    # ----------------------- aggregate ------------------------------- #
    summed = (x * mask_expanded).sum(dim=1)
    denom = mask_expanded.sum(dim=1).clamp(min=1e-6)  # avoid div/0 for all-padded
    feats = summed / denom
    return feats