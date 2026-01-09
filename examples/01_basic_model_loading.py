"""
Example 1: Basic Model Loading

This example demonstrates the fundamental model loading capabilities:
- Loading pre-trained models with weights
- Creating new models for training
- Using class mappings for predictions
- Using different model configurations

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

from avex import describe_model, get_model_spec, list_models, load_model
from avex.configs import AudioConfig, ModelSpec, ProbeConfig
from avex.models.probes.utils import build_probe_from_config
from avex.models.utils.factory import build_model_from_spec


def main(device: str = "cpu") -> None:
    """Demonstrate basic model loading capabilities."""
    print("Example 1: Basic Model Loading")
    print("=" * 50)

    # =========================================================================
    # Part 1: List available models
    # =========================================================================
    print("\nPart 1: Available Models")
    print("-" * 50)
    list_models()  # Prints formatted table

    # =========================================================================
    # Part 2: Load a pre-trained model with checkpoint
    # =========================================================================
    print("\nPart 2: Load Pre-trained Model with Checkpoint")
    print("-" * 50)

    # Load model with checkpoint - num_classes extracted automatically
    model = load_model("efficientnet_animalspeak", device=device)
    print(f"Loaded model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of classes: {output.shape[-1]}")

    # Use class mapping for predictions if available
    if hasattr(model, "label_mapping") and model.label_mapping:
        index_to_label = model.label_mapping["index_to_label"]
        label_to_index = model.label_mapping["label_to_index"]
        print(f"\n   Label mapping available: {len(label_to_index)} classes")

        # Get top-3 predictions with actual class labels
        probs = torch.softmax(output, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(3, output.shape[-1]), dim=-1)
        print("   Top-3 predicted classes:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0], strict=False)):
            label = index_to_label.get(idx.item(), f"Unknown (index {idx.item()})")
            print(f"     {i + 1}. {label}: {prob.item():.4f}")

    # =========================================================================
    # Part 3: Create a new model for training
    # =========================================================================
    print("\nPart 3: Create New Model for Training")
    print("-" * 50)

    model_spec = get_model_spec("sl_beats_animalspeak")

    # Build backbone-only model from spec
    backbone = build_model_from_spec(model_spec, device=device).to(device)
    backbone.eval()

    # Attach a linear probe for a 50-class task
    probe_cfg = ProbeConfig(
        probe_type="linear",
        target_layers=["last_layer"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    model = build_probe_from_config(
        probe_config=probe_cfg,
        base_model=backbone,
        num_classes=50,
        device=device,
    ).to(device)

    print(f"Created backbone: {type(backbone).__name__}")
    print(f"Created probe model: {type(model).__name__}")
    print(f"   Parameters (backbone + probe): {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 16000 * 3, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # =========================================================================
    # Part 4: Load model with custom parameters
    # =========================================================================
    print("\nPart 4: Load Model with Custom Parameters")
    print("-" * 50)

    custom_spec = ModelSpec(
        name="efficientnet",
        pretrained=False,
        device=device,
        audio_config=AudioConfig(
            sample_rate=16000,
            representation="mel_spectrogram",
            n_mels=128,
            target_length_seconds=5,
        ),
        efficientnet_variant="b1",
    )

    # Build backbone-only model from custom spec
    backbone_custom = build_model_from_spec(custom_spec, device=device).to(device)
    backbone_custom.eval()

    probe_cfg_custom = ProbeConfig(
        probe_type="linear",
        target_layers=["last_layer"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    model = build_probe_from_config(
        probe_config=probe_cfg_custom,
        base_model=backbone_custom,
        num_classes=25,
        device=device,
    ).to(device)

    print(f"Created model with custom parameters: {type(model).__name__}")
    print("   Variant: b1")
    print(f"   Parameters (backbone + probe): {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # Part 5: Get detailed model information
    # =========================================================================
    print("\nPart 5: Model Information")
    print("-" * 50)
    describe_model("beats_naturelm", verbose=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("""
- load_model(): Loads full models or backbones, extracts num_classes automatically when checkpoints include classifiers
- build_model_from_spec(): Creates backbone models from ModelSpec via the registry
- Probes (build_probe_from_config): Attach task-specific heads with arbitrary num_classes
- label_mapping: Available for models with trained classifiers
- Custom ModelSpec: Override backbone architecture parameters for specific needs
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Model Loading Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
