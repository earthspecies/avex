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
    load_class_mapping,
    load_model,
)


def main() -> None:
    print("🚀 Example 2: Checkpoint Loading and Management")
    print("=" * 50)

    # Example 1: View default checkpoints from YAML
    print("\n📋 Default checkpoints from YAML configurations:")
    print("   Note: Checkpoint paths are defined in YAML files in configs/official_models/")

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
        # Use get_model_spec directly to avoid plugin architecture issues
        from representation_learning import get_model_spec
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
        model = load_model("beats_naturelm", checkpoint_path=str(dummy_checkpoint_path), device="cpu")
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

    # Example 7: Class mapping for models with classifier heads
    print("\n🏷️  Class Mapping for Models with Classifier Heads:")
    print("   Note: Some models have classifier heads where logits correspond to specific class labels")
    print("   The class mapping is defined in YAML files (class_mapping_path)")
    try:
        # Show which models have class mappings by trying to load them
        models_with_mapping = []
        for model_name in list_models().keys():
            class_mapping = load_class_mapping(model_name)
            if class_mapping:
                models_with_mapping.append(model_name)

        if models_with_mapping:
            print(f"\n   Models with class mappings ({len(models_with_mapping)}):")
            for model_name in models_with_mapping:
                print(f"     - {model_name}")
        else:
            print("   No models with class mappings found")

        # Example: Load a model with class mapping
        print("\n   Example: Loading model with class mapping:")
        model_name = "sl_beats_animalspeak"
        print(f"   Loading class mapping for model: {model_name}")
        class_mapping = load_class_mapping(model_name)
        if class_mapping:
            label_to_index = class_mapping["label_to_index"]
            index_to_label = class_mapping["index_to_label"]
            print(f"   ✅ Loaded class mapping with {len(label_to_index)} classes")
            print("   Example labels (first 5):")
            for _i, (label, idx) in enumerate(list(label_to_index.items())[:5]):
                print(f"     - {label}: index {idx}")
            print("   Example reverse mapping (indices 0-4):")
            for idx in range(min(5, len(index_to_label))):
                if idx in index_to_label:
                    print(f"     - index {idx}: {index_to_label[idx]}")

            # Demonstrate loading model with automatic class mapping attachment
            print(f"\n   Loading model '{model_name}' (class mapping will be attached automatically):")
            try:
                model = load_model(model_name, device="cpu")
                if hasattr(model, "class_mapping"):
                    print("   ✅ Model loaded with class mapping attached")
                    print(
                        "   Access via: model.class_mapping['label_to_index'] or model.class_mapping['index_to_label']"
                    )

                    # Example: Use class mapping for predictions
                    print("\n   Example: Using class mapping for predictions:")
                    dummy_input = torch.randn(1, 16000 * 5)
                    with torch.no_grad():
                        logits = model(dummy_input)
                    # Get top-3 predictions
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, k=min(3, logits.shape[-1]), dim=-1)
                    print("   Top-3 predicted classes:")
                    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0], strict=False)):
                        idx_int = idx.item()
                        label = index_to_label.get(idx_int, f"Unknown (index {idx_int})")
                        print(f"     {i + 1}. {label}: {prob.item():.4f}")
                else:
                    print("   ⚠️  Model loaded but class mapping not attached (checkpoint may not exist)")
            except Exception as e:
                print(f"   ⚠️  Could not load model (checkpoint may not exist): {e}")
        else:
            print("   ⚠️  Could not load class mapping (file may not exist or model has no mapping defined)")

    except Exception as e:
        print(f"❌ Error working with class mappings: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
