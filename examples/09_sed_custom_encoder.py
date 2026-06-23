"""
Example 9: Register a SED BEATs checkpoint as a custom encoder

This example shows how to take the frame-level sound-event-detection (SED) model
trained in the `sound-event-detection` project and register it as a *custom model*
in avex, so it can be:

  1. Loaded by name through the standard avex API, and
  2. Reused as an **encoder** when building a larger model.

Checkpoint:
    gs://sound-event-detection/checkpoints/final_ssl_beats_clip_pseudo_ensA_040/best_model.pt

Architecture
------------
The SED model is a BEATs SSL backbone whose frame-level patch embeddings are
concatenated over the frequency dimension, followed by a per-frame linear head:

    waveform [B, samples]
        -> BEATs backbone        -> [B, T * F, 768]   (F = 8 frequency patches)
        -> concat over frequency -> [B, T, 768 * 8]    (= [B, T, 6144])
        -> linear classifier     -> [B, T, num_classes]

The internal attribute layout (`encoder._model.backbone.*` / `classifier.*`)
mirrors the structure used to train the checkpoint, so the original
`model_state_dict` loads with no key remapping.

Run:
    python examples/09_sed_custom_encoder.py --device cpu
"""

import argparse
from typing import Optional

import torch
import torch.nn as nn

from avex import load_model, register_model, register_model_class
from avex.configs import AudioConfig, ModelSpec
from avex.models.base_model import ModelBase
from avex.models.beats_model import Model as BeatsModel

# Registry name and checkpoint location for the SED model.
SED_MODEL_NAME = "sed_beats_clip_pseudo_ensA_040"
SED_CHECKPOINT = "gs://sound-event-detection/checkpoints/final_ssl_beats_clip_pseudo_ensA_040/best_model.pt"

# BEATs hidden size and number of frequency patches per time step. Concatenating
# the frequency patches gives an output dim of BEATS_HIDDEN_DIM * NUM_FREQ_PATCHES.
BEATS_HIDDEN_DIM = 768
NUM_FREQ_PATCHES = 8

# BEATs fbank constants. BEATs' fbank is configured with sample_frequency=16000
# (its default) regardless of the real input rate. The SED model feeds 32 kHz audio
# un-resampled, so these 16 kHz-derived window/hop sizes (400/160 samples) are applied
# directly to the 32 kHz stream — matching how the checkpoint was trained.
_FBANK_WINDOW_SAMPLES = 400
_FBANK_HOP_SAMPLES = 160
_PATCH_SIZE = 16

# BEATs iter3+ AS2M SSL backbone configuration (matches the trained checkpoint).
SSL_BEATS_CONFIG = {
    "activation_dropout": 0.0,
    "activation_fn": "gelu",
    "attention_dropout": 0.1,
    "conv_bias": False,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "deep_norm": True,
    "dropout": 0.1,
    "dropout_input": 0.1,
    "embed_dim": 512,
    "encoder_attention_heads": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_layerdrop": 0.05,
    "encoder_layers": 12,
    "finetuned_model": False,
    "gru_rel_pos": True,
    "input_patch_size": 16,
    "layer_norm_first": False,
    "layer_wise_gradient_decay_ratio": 1.0,
    "max_distance": 800,
    "num_buckets": 320,
    "relative_position_embedding": True,
}


def compute_beats_framerate(sample_rate: int, window_duration: float) -> float:
    """Compute the BEATs output framerate (Hz) for a given sample rate / window.

    Returns
    -------
    float
        Output framerate in Hz.
    """
    window_samples = int(window_duration * sample_rate)
    t_fbank = (window_samples - _FBANK_WINDOW_SAMPLES) // _FBANK_HOP_SAMPLES + 1
    t_patched = t_fbank // _PATCH_SIZE
    return t_patched / window_duration


class _BEATSConcatEncoder(nn.Module):
    """BEATs backbone that concatenates frequency patches into frame embeddings.

    The wrapped avex BEATs model is stored as ``self._model`` (matching the
    checkpoint key layout ``encoder._model.backbone.*``) and always runs in
    ``return_features_only`` mode.
    """

    def __init__(self, *, device: str, init_config: Optional[dict] = None) -> None:
        super().__init__()
        # Feed raw waveforms straight to BEATs (audio_config=None => no extra
        # preprocessing); BEATs applies its own fbank frontend internally.
        self._model = BeatsModel(
            num_classes=None,
            pretrained=False,
            device=device,
            audio_config=None,
            return_features_only=True,
            init_config=init_config,
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a waveform into concatenated frame embeddings [B, T, 6144].

        Returns
        -------
        torch.Tensor
            Frame embeddings of shape ``(batch, time, 768 * 8)``.
        """
        embeddings = self._model(x, padding_mask=None)  # [B, T * F, 768]
        batch_size, _, hidden_dim = embeddings.shape
        return embeddings.reshape(batch_size, -1, NUM_FREQ_PATCHES, hidden_dim).reshape(
            batch_size, -1, NUM_FREQ_PATCHES * hidden_dim
        )

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self._model.parameters():
            param.requires_grad = True


@register_model_class
class Model(ModelBase):
    """Frame-level BEATs sound-event-detection model (registered as ``sed_beats``).

    Wraps a BEATs SSL backbone with frequency-concatenation aggregation and an
    optional per-frame linear classifier. Follows avex model conventions so it
    can be built via the registry and reused as an encoder for larger models.

    When ``num_classes is None`` or ``return_features_only=True`` the classifier
    is omitted and ``forward`` returns frame embeddings of shape
    ``(batch, time, 6144)``; otherwise it returns per-frame logits.
    """

    name = "sed_beats"

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        init_config: Optional[dict] = None,
        sample_rate: int = 32000,
        window_duration: float = 5.0,
    ) -> None:
        # Keep audio passthrough (audio_config=None on the base): BEATs handles
        # its own frontend, so we must not resample the raw waveform here.
        super().__init__(device=device, audio_config=None)

        if num_classes is None:
            return_features_only = True
        self.num_classes = num_classes

        self._return_features_only = return_features_only
        self._sample_rate = sample_rate
        self._window_duration = window_duration

        self.encoder = _BEATSConcatEncoder(device=device, init_config=init_config)

        if not return_features_only and num_classes is not None:
            self.classifier: Optional[nn.Module] = nn.Linear(self.output_dim, num_classes)
        else:
            self.register_module("classifier", None)  # type: ignore[arg-type]

        self.to(device)

    @property
    def output_dim(self) -> int:
        """Dimension of the frame embeddings produced by the encoder."""
        return BEATS_HIDDEN_DIM * NUM_FREQ_PATCHES

    @property
    def output_framerate(self) -> float:
        """Output framerate in Hz for the configured sample rate / window."""
        return compute_beats_framerate(self._sample_rate, self._window_duration)

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate in Hz."""
        return self._sample_rate

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return frame embeddings or per-frame logits.

        Returns
        -------
        torch.Tensor
            ``(batch, time, 6144)`` embeddings when in feature mode, else
            ``(batch, time, num_classes)`` logits.
        """
        embeddings = self.encoder(x)
        if self._return_features_only or self.classifier is None:
            return embeddings
        return self.classifier(embeddings)

    def freeze_encoder(self) -> None:
        """Freeze the BEATs encoder parameters."""
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """Unfreeze the BEATs encoder parameters."""
        self.encoder.unfreeze()


def register_sed_model() -> None:
    """Register a ModelSpec for the SED checkpoint under ``SED_MODEL_NAME``.

    The model class is registered via the ``@register_model_class`` decorator on
    ``Model``; here we register the matching ``ModelSpec`` so the checkpoint can
    be loaded by name with ``load_model``.
    """
    spec = ModelSpec(
        name="sed_beats",
        pretrained=False,
        device="cpu",
        init_config=SSL_BEATS_CONFIG,
        audio_config=AudioConfig(
            sample_rate=32000,
            representation="raw",
            normalize=False,
            target_length_seconds=5,
        ),
    )
    register_model(SED_MODEL_NAME, spec)


class LargerModel(nn.Module):
    """Example larger model built on top of the frozen SED encoder.

    Mean-pools the frame embeddings over time and applies a new classification
    head — a stand-in for whatever downstream architecture you want to build.
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, mean-pool over time, then classify.

        Returns
        -------
        torch.Tensor
            Clip-level logits of shape ``(batch, num_classes)``.
        """
        frame_embeddings = self.encoder(x)  # [B, T, 6144]
        pooled = frame_embeddings.mean(dim=1)  # [B, 6144]
        return self.head(pooled)


def main(device: str = "cpu") -> None:
    """Register the SED model and demonstrate using it as an encoder."""
    print("Example 9: SED BEATs as a custom encoder")
    print("=" * 60)

    register_sed_model()
    print(f"Registered custom model: {SED_MODEL_NAME}")

    # ------------------------------------------------------------------
    # 1. Load as an encoder (classifier stripped, frame embeddings out).
    # ------------------------------------------------------------------
    print("\n[1] Load as encoder (return_features_only=True)")
    encoder = load_model(
        SED_MODEL_NAME,
        device=device,
        checkpoint_path=SED_CHECKPOINT,
        return_features_only=True,
    )
    encoder.eval()

    waveform = torch.randn(2, int(5.0 * 32000), device=device)  # 2 x 5s @ 32kHz
    with torch.no_grad():
        frame_embeddings = encoder(waveform)
    print(f"    output_dim:       {encoder.output_dim}")
    print(f"    output_framerate: {encoder.output_framerate:.2f} Hz")
    print(f"    frame embeddings: {tuple(frame_embeddings.shape)}")

    # ------------------------------------------------------------------
    # 2. Build a larger model on top of the frozen encoder.
    # ------------------------------------------------------------------
    print("\n[2] Build a larger model on top of the encoder")
    encoder.freeze_encoder()
    larger = LargerModel(encoder=encoder, embedding_dim=encoder.output_dim, num_classes=10).to(device)
    with torch.no_grad():
        clip_logits = larger(waveform)
    trainable = sum(p.numel() for p in larger.parameters() if p.requires_grad)
    print(f"    clip logits:        {tuple(clip_logits.shape)}")
    print(f"    trainable params:   {trainable:,} (encoder frozen)")

    # ------------------------------------------------------------------
    # 3. (Optional) Load the full frame-level detector with its trained head.
    # ------------------------------------------------------------------
    print("\n[3] Load the full frame-level detector (trained classifier)")
    detector = load_model(SED_MODEL_NAME, device=device, checkpoint_path=SED_CHECKPOINT)
    detector.eval()
    with torch.no_grad():
        frame_logits = detector(waveform)
    print(f"    num_classes:  {detector.num_classes}")
    print(f"    frame logits: {tuple(frame_logits.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED custom encoder example")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu or cuda")
    args = parser.parse_args()
    main(device=args.device)
