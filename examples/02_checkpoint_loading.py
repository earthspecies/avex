"""
Example 2: Checkpoint Loading and Management

This example demonstrates:
- Loading models with default checkpoints (from YAML)
- Loading models with custom checkpoints
- Extracting num_classes from checkpoints
- Working with cloud storage paths

IMPORTANT: Model specifications (architecture, parameters) and default checkpoint paths
are defined in YAML files in configs/official_models/. Checkpoint paths can be
overridden by passing checkpoint_path parameter to load_model().
"""

import torch

from representation_learning import (
    get_checkpoint_path,
    list_models,
    load_model,
)


def main() -> None:
    print("🚀 Example 2: Checkpoint Loading and Management")
    print("=" * 50)

    # Example 1: View default checkpoints from YAML
    print("\n📋 Default checkpoints from YAML configurations:")
    print(
        "   Note: Checkpoint paths are defined in YAML files in "
        "configs/official_models/"
    )

    try:
        # List registered models and their default checkpoint paths
        models = list_models()
        for model_name in models.keys():
            checkpoint = get_checkpoint_path(model_name)
            if checkpoint:
                print(f"  - {model_name}: {checkpoint}")
            else:
                print(f"  - {model_name}: No default checkpoint (model spec only)")

    except Exception as e:
        print(f"❌ Error listing checkpoints: {e}")

    # Example 2: Load model with default checkpoint
    print("\n🔧 Loading model with default checkpoint:")
    try:
        # Use get_model directly to avoid plugin architecture issues
        from representation_learning import get_model as get_model_spec
        from representation_learning.models.get_model import get_model

        model_spec = get_model_spec("efficientnet_animalspeak")
        model = get_model(model_spec, num_classes=10)
        model = model.cpu()
        print(f"✅ Loaded model with default checkpoint: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error loading with default checkpoint: {e}")
        print("   (This is expected if the checkpoint doesn't exist)")

    # Example 3: Load model with custom checkpoint
    print("\n🔧 Loading model with custom checkpoint:")
    try:
        # Create a dummy checkpoint for demonstration
        from pathlib import Path

        # Ensure checkpoints directory exists
        checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        dummy_checkpoint_path = checkpoints_dir / "dummy_checkpoint.pt"
        dummy_state_dict = {
            "classifier.weight": torch.randn(15, 768),  # 15 classes
            "classifier.bias": torch.randn(15),
        }
        torch.save(dummy_state_dict, dummy_checkpoint_path)

        # Load with custom checkpoint
        model = load_model(
            "beats_naturelm", checkpoint_path=str(dummy_checkpoint_path), device="cpu"
        )
        print(f"✅ Loaded model with custom checkpoint: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print(f"   Extracted num_classes: {output.shape[-1]}")

        # Note: Checkpoint saved to checkpoints/ directory for future use
        print(f"   Checkpoint saved to: {dummy_checkpoint_path}")
        # Optionally clean up:
        # dummy_checkpoint_path.unlink()

    except Exception as e:
        print(f"❌ Error loading with custom checkpoint: {e}")

    # Example 4: Load model with explicit num_classes (overrides checkpoint)
    print("\n🔧 Loading model with explicit num_classes:")
    try:
        model = load_model(
            "efficientnet_animalspeak",
            num_classes=20,  # Override any checkpoint num_classes
            device="cpu",
        )
        print(f"✅ Loaded model with explicit num_classes: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(2, 16000 * 3)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error loading with explicit num_classes: {e}")

    # Example 5: Working with cloud storage paths
    print("\n🔧 Working with cloud storage paths:")
    try:
        # Load from Google Cloud Storage path
        model = load_model(
            "sl_eat_animalspeak_ssl_all",
            checkpoint_path="gs://representation-learning/models/sl_eat_animalspeak_ssl_all.pt",
            device="cpu",
        )
        print(f"✅ Loaded model from cloud storage: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ Error loading from cloud storage: {e}")
        print("   (This is expected if the cloud checkpoint doesn't exist)")

    # Example 6: Checkpoint management
    print("\n📊 Checkpoint Information:")
    try:
        # Get checkpoint info from YAML
        checkpoint = get_checkpoint_path("efficientnet_animalspeak")
        print(f"Default checkpoint for efficientnet_animalspeak: {checkpoint}")
        print("   Note: Checkpoint paths are read from YAML files")
        print("   To override, pass checkpoint_path parameter to load_model()")

    except Exception as e:
        print(f"❌ Error getting checkpoint info: {e}")


if __name__ == "__main__":
    main()
