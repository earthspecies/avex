"""
Example 2: Checkpoint Loading and Management

This example demonstrates:
- Loading models with default checkpoints (from YAML)
- Loading models with custom checkpoints
- Extracting num_classes from checkpoints
- Working with class mappings

IMPORTANT: Model specifications and default checkpoint paths are defined in
YAML files in configs/official_models/. Checkpoint paths can be overridden
by passing checkpoint_path parameter to load_model().

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
from pathlib import Path

import torch

from representation_learning import (
    get_checkpoint_path,
    get_model_spec,
    list_models,
    load_label_mapping,
    load_model,
)
from representation_learning.models.get_model import get_model


def main(device: str = "cpu") -> None:
    """Demonstrate checkpoint loading and management."""
    print("Example 2: Checkpoint Loading and Management")
    print("=" * 50)

    # =========================================================================
    # Part 1: View default checkpoints from YAML
    # =========================================================================
    print("\nPart 1: Default Checkpoints from YAML")
    print("-" * 50)
    print("Note: Checkpoint paths are defined in configs/official_models/*.yml\n")

    models = list_models()

    print("Checkpoint details:")
    for model_name, info in models.items():
        checkpoint = info.get("checkpoint_path")
        if checkpoint:
            display = checkpoint[:50] + "..." if len(checkpoint) > 50 else checkpoint
            print(f"  - {model_name}: {display}")
        else:
            print(f"  - {model_name}: No checkpoint")

    # =========================================================================
    # Part 2: Load model with default checkpoint
    # =========================================================================
    print("\nPart 2: Load Model with Default Checkpoint")
    print("-" * 50)

    model_spec = get_model_spec("efficientnet_animalspeak")
    model = get_model(model_spec, num_classes=10)
    model = model.to(device)

    print(f"Loaded model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Output shape: {output.shape}")

    # =========================================================================
    # Part 3: Load model with custom checkpoint
    # =========================================================================
    print("\nPart 3: Load Model with Custom Checkpoint")
    print("-" * 50)

    # Ensure checkpoints directory exists
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Create a dummy checkpoint for demonstration
    dummy_checkpoint_path = checkpoints_dir / "dummy_checkpoint.pt"
    dummy_state_dict = {
        "classifier.weight": torch.randn(15, 768),  # 15 classes
        "classifier.bias": torch.randn(15),
    }
    torch.save(dummy_state_dict, dummy_checkpoint_path)
    print(f"Created dummy checkpoint: {dummy_checkpoint_path}")

    # Load with custom checkpoint
    model = load_model("beats_naturelm", checkpoint_path=str(dummy_checkpoint_path), device=device)
    print(f"Loaded model: {type(model).__name__}")

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Extracted num_classes: {output.shape[-1]}")

    # =========================================================================
    # Part 4: Load model with explicit num_classes
    # =========================================================================
    print("\nPart 4: Load Model with Explicit num_classes")
    print("-" * 50)
    print("Note: Explicit num_classes creates a new random classifier (ignores checkpoint classifier)")

    model = load_model("efficientnet_animalspeak", num_classes=20, device=device)
    print(f"Loaded model: {type(model).__name__}")

    dummy_input = torch.randn(2, 16000 * 3, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   num_classes: {output.shape[-1]}")

    # =========================================================================
    # Part 5: Class mapping for models with classifier heads
    # =========================================================================
    print("\nPart 5: Class Mapping")
    print("-" * 50)

    model_name = "sl_beats_animalspeak"
    label_mapping = load_label_mapping(model_name)

    if label_mapping:
        label_to_index = label_mapping["label_to_index"]
        index_to_label = label_mapping["index_to_label"]
        print(f"Loaded class mapping for '{model_name}': {len(label_to_index)} classes")

        print("\nExample labels (first 5):")
        for label, idx in list(label_to_index.items())[:5]:
            print(f"  - {label}: index {idx}")

        print("\nExample reverse mapping (indices 0-4):")
        for idx in range(min(5, len(index_to_label))):
            print(f"  - index {idx}: {index_to_label.get(idx, 'N/A')}")

    # =========================================================================
    # Part 6: Checkpoint information utility
    # =========================================================================
    print("\nPart 6: Checkpoint Information")
    print("-" * 50)

    checkpoint = get_checkpoint_path("efficientnet_animalspeak")
    print("Default checkpoint for efficientnet_animalspeak:")
    print(f"   {checkpoint}")
    print("\nTo override: load_model(name, checkpoint_path='your/path.pt')")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("""
- Default checkpoints defined in YAML files
- Use checkpoint_path parameter to override
- num_classes=None: Extract from checkpoint
- num_classes=N: Create new classifier (random init)
- Class mappings link logit indices to labels
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint Loading Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
