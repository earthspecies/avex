"""
Example 2: Checkpoint Loading and Management

This example demonstrates:
- Loading models with custom checkpoints
- Registering default checkpoints
- Extracting num_classes from checkpoints
- Working with cloud storage paths

IMPORTANT: Model specifications (architecture, parameters) are defined in YAML files
in configs/official_models/, while checkpoint paths (pre-trained weights) are registered
separately using register_checkpoint(). This separation allows for flexible model
loading with different checkpoint sources.
"""

import torch

from representation_learning import (
    get_checkpoint,
    list_models,
    load_model,
    register_checkpoint,
    unregister_checkpoint,
)


def main() -> None:
    print("🚀 Example 2: Checkpoint Loading and Management")
    print("=" * 50)

    # Example 1: Register default checkpoints
    print("\n📋 Registering default checkpoints:")
    print(
        "   Note: Model specs define architecture, checkpoints define "
        "pre-trained weights"
    )
    print(
        "   These are registered separately - model specs come from YAML, "
        "checkpoints from registry"
    )

    try:
        # Register checkpoints for different models
        # These checkpoint paths are separate from the model specifications in
        # YAML files
        # Register real checkpoint paths from Google Cloud Storage
        register_checkpoint(
            "beats_naturelm", "gs://representation-learning/models/beats_naturelm.pt"
        )
        register_checkpoint(
            "efficientnet_animalspeak",
            "gs://representation-learning/models/efficientnet_animalspeak.pt",
        )
        register_checkpoint(
            "sl_eat_animalspeak_ssl_all",
            "gs://representation-learning/models/sl_eat_animalspeak_ssl_all.pt",
        )

        print("✅ Registered default checkpoints")

        # List registered checkpoints
        models = list_models()
        for model_name in models.keys():
            checkpoint = get_checkpoint(model_name)
            if checkpoint:
                print(f"  - {model_name}: {checkpoint}")
            else:
                print(f"  - {model_name}: No default checkpoint (model spec only)")

    except Exception as e:
        print(f"❌ Error registering checkpoints: {e}")

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
        dummy_checkpoint_path = "dummy_checkpoint.pt"
        dummy_state_dict = {
            "classifier.weight": torch.randn(15, 768),  # 15 classes
            "classifier.bias": torch.randn(15),
        }
        torch.save(dummy_state_dict, dummy_checkpoint_path)

        # Load with custom checkpoint
        model = load_model(
            "beats_naturelm", checkpoint_path=dummy_checkpoint_path, device="cpu"
        )
        print(f"✅ Loaded model with custom checkpoint: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print(f"   Extracted num_classes: {output.shape[-1]}")

        # Clean up dummy checkpoint
        import os

        os.remove(dummy_checkpoint_path)

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
    print("\n📊 Checkpoint Management:")
    try:
        # Get checkpoint info
        checkpoint = get_checkpoint("beats_naturelm")
        print(f"Default checkpoint for beats_naturelm: {checkpoint}")

        # Unregister a checkpoint
        unregister_checkpoint("efficientnet_animalspeak")
        print("✅ Unregistered efficientnet_animalspeak checkpoint")

        # Check if it's gone
        checkpoint = get_checkpoint("efficientnet_animalspeak")
        print(f"Checkpoint after unregistering: {checkpoint}")

    except Exception as e:
        print(f"❌ Error in checkpoint management: {e}")


if __name__ == "__main__":
    main()
