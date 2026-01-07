"""
Quick Start Example

This example demonstrates the basic functionality of the representation-learning library:
- Listing available models
- Getting model information
- Loading and running a model

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: describe_model("model_name") or get_model_spec("model_name").audio_config.sample_rate
- For full reproducibility, resample using librosa with these exact parameters:

    import librosa
    audio_resampled = librosa.resample(
        audio, orig_sr=original_sr, target_sr=target_sr,
        res_type="kaiser_best", scale=True
    )
"""

import argparse

import torch

from representation_learning import describe_model, get_model_spec, list_models
from representation_learning.configs import ProbeConfig
from representation_learning.models.probes.utils import build_probe_from_config
from representation_learning.models.utils.factory import build_model_from_spec


def main(device: str = "cpu") -> None:
    """Demonstrate basic library functionality."""
    print("Quick Start Example")
    print("=" * 50)

    # =========================================================================
    # Part 1: List available models
    # =========================================================================
    print("\nPart 1: Available Models")
    print("-" * 50)
    models = list_models()  # Prints formatted table
    print(f"\nTotal available models: {len(models)}")

    # =========================================================================
    # Part 2: Get model information
    # =========================================================================
    print("\nPart 2: Model Information")
    print("-" * 50)

    # Use beats_naturelm as example
    model_name = "beats_naturelm"
    print(f"Detailed information for '{model_name}':")
    describe_model(model_name, verbose=True)

    # =========================================================================
    # Part 3: Load and test a model
    # =========================================================================
    print("\nPart 3: Load and Test Model")
    print("-" * 50)

    model_spec = get_model_spec(model_name)

    # Build backbone-only model
    backbone = build_model_from_spec(model_spec, device=device).to(device)
    backbone.eval()

    # Attach a simple linear probe for a 10-class task
    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["backbone"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    model = build_probe_from_config(
        probe_config=probe_config,
        base_model=backbone,
        num_classes=10,
        device=device,
    ).to(device)
    model.eval()

    print(f"Created backbone: {type(backbone).__name__}")
    print(f"Created probe model: {type(model).__name__}")
    print(f"   Total parameters (backbone + probe): {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass (BEATs expects 16kHz audio)
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Functions")
    print("=" * 50)
    print("""
- list_models(): List all available models with details
- describe_model(name): Get detailed model information
- get_model_spec(name): Get model specification
- build_model_from_spec(spec, device): Create backbone model from ModelSpec via registry
- build_probe_from_config(...): Attach task-specific heads in online or offline mode
- load_model(name, ...): Load full models or backbones with optional checkpoint
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Start Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (default: cpu)",
    )
    args = parser.parse_args()
    main(device=args.device)
