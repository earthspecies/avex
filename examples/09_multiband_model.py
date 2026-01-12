"""
Example 9: Multiband Audio Processing

This example demonstrates the multiband wrapper that adds high sample rate
support to ANY existing audio backbone:

1. MultibandWrapper - wraps any backbone for multiband processing
2. MultibandTransform - splits audio into frequency bands
3. BandScorer - computes informativeness scores per band
"""

import argparse

import torch
import torch.nn as nn

from representation_learning.models.multiband import (
    BandScorer,
    MultibandTransform,
    MultibandWrapper,
)


def main(device: str = "cpu") -> None:
    print("Example 9: Multiband Audio Processing")
    print("=" * 60)

    # =========================================================================
    # Part 1: MultibandWrapper with a simple backbone
    # =========================================================================
    print("\nPart 1: MultibandWrapper with custom backbone")
    print("-" * 60)

    # Create a simple backbone (simulates any audio model)
    # In practice, you'd use: load_model("beats_naturelm", device=device)
    class SimpleBackbone(nn.Module):
        """Dummy backbone that simulates an audio encoder."""
        def __init__(self, embed_dim=256):
            super().__init__()
            self.embed_dim = embed_dim
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=40),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, embed_dim),
            )

        def forward(self, x):
            # x: (batch, time) -> (batch, embed_dim)
            return self.encoder(x.unsqueeze(1))

    backbone = SimpleBackbone(embed_dim=256).to(device)
    print(f"Backbone: SimpleBackbone, embed_dim=256")
    print(f"Backbone expects: 16kHz input")

    # Wrap it for multiband processing
    model = MultibandWrapper(
        backbone=backbone,
        sample_rate=44100,      # Input sample rate
        baseband_sr=16000,      # What backbone expects
        band_width_hz=8000,     # 8kHz per band
        fusion_type="attention",
        embed_dim=256,
    )
    model.eval()

    print(f"\nMultibandWrapper:")
    print(f"  Input sample rate: 44100 Hz")
    print(f"  Bands: {model.num_bands}")
    for i, (f_low, f_high) in enumerate(model.get_band_info()):
        print(f"    Band {i}: {f_low/1000:.1f} - {f_high/1000:.1f} kHz")
    print(f"  Fusion: attention")

    # Forward pass with high sample rate audio
    audio_44k = torch.randn(2, 44100 * 3, device=device)  # 3 sec at 44.1kHz
    with torch.no_grad():
        embeddings = model(audio_44k)

    print(f"\n  Input: {audio_44k.shape} (44.1kHz)")
    print(f"  Output: {embeddings.shape} (fused embeddings)")

    # =========================================================================
    # Part 2: Different fusion types
    # =========================================================================
    print("\nPart 2: Different fusion types")
    print("-" * 60)

    for fusion_type in ["attention", "gated", "concat"]:
        model = MultibandWrapper(
            backbone=SimpleBackbone(embed_dim=256).to(device),
            sample_rate=44100,
            fusion_type=fusion_type,
            embed_dim=256,
        )
        model.eval()

        with torch.no_grad():
            out = model(audio_44k)

        weights = model.get_band_weights()
        print(f"{fusion_type:12s}: output={out.shape}, weights={[f'{w:.3f}' for w in weights.tolist()]}")

    # =========================================================================
    # Part 3: With classification head
    # =========================================================================
    print("\nPart 3: With classification head")
    print("-" * 60)

    model = MultibandWrapper(
        backbone=SimpleBackbone(embed_dim=256).to(device),
        sample_rate=44100,
        fusion_type="gated",
        embed_dim=256,
        num_classes=10,  # Add classifier
    )
    model.eval()

    with torch.no_grad():
        logits = model(audio_44k)

    print(f"Input: {audio_44k.shape}")
    print(f"Output: {logits.shape} (class logits)")

    # =========================================================================
    # Part 4: With a real backbone (conceptual)
    # =========================================================================
    print("\nPart 4: With real backbones (conceptual)")
    print("-" * 60)
    print("""
# How to use with real models:

from representation_learning import load_model
from representation_learning.models.multiband import MultibandWrapper

# Load any backbone
backbone = load_model("beats_naturelm", device="cuda")
# OR: backbone = load_model("sl_eat_animalspeak_ssl_all", device="cuda")
# OR: backbone = load_model("efficientnet", device="cuda")

# Wrap for multiband
model = MultibandWrapper(
    backbone=backbone,
    sample_rate=96000,      # Your high sample rate
    baseband_sr=16000,      # What backbone expects
    band_width_hz=8000,
    fusion_type="attention",
)

# Now handles 96kHz input!
audio_96k = load_audio("dolphin_clicks.wav")  # 96kHz
embeddings = model(audio_96k)
""")

    # =========================================================================
    # Part 5: Band Scoring
    # =========================================================================
    print("Part 5: Band Scoring")
    print("-" * 60)

    import torchaudio

    scorer = BandScorer(
        sample_rate=44100,
        band_width_hz=8000,
        score_types=["entropy", "flux"],
    )

    # Create spectrogram for scoring
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=1024, hop_length=512, n_mels=128,
    )
    spec = torch.log(mel_transform(audio_44k.cpu()) + 1e-8)

    scores = scorer(spec)
    print(f"Entropy per band: {[f'{s:.2f}' for s in scores.entropy.mean(dim=0).tolist()]}")
    print(f"Flux per band: {[f'{s:.0f}' for s in scores.flux.mean(dim=0).tolist()]}")

    top_bands = scorer.select_top_k(spec, k=2, method="entropy")
    print(f"Top 2 bands by entropy: {top_bands}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
MultibandWrapper adds high sample rate support to ANY backbone:

1. Takes any existing model (BEATs, EAT, EfficientNet, custom)
2. Splits input audio into frequency bands via heterodyning
3. Runs each band through the backbone
4. Fuses per-band embeddings (attention/gated/concat)

Usage:
    wrapper = MultibandWrapper(
        backbone=your_model,      # Any audio model
        sample_rate=96000,        # Your input sample rate
        baseband_sr=16000,        # What backbone expects
        fusion_type="attention",
    )
    output = wrapper(high_sr_audio)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiband Wrapper Example")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(device=args.device)
