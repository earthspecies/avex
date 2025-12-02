"""
Example 1: Basic Model Loading

This example demonstrates the fundamental model loading capabilities:
- Loading pre-trained models with weights
- Creating new models for training
- Using different model types and configurations
- Using class mappings for predictions
"""

import torch

from representation_learning import describe_model, get_model_spec, list_models, load_model
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Example 1: Basic Model Loading")
    print("=" * 50)

    # List available models (prints formatted table and returns info dict)
    print("\n📋 Available Models:")
    list_models()  # Prints formatted table
    # Note: list_models() automatically prints a formatted table above
    # The returned dict contains detailed info for programmatic access

    # Example 1: Load a pre-trained model with checkpoint and class mapping
    print("\n🔧 Loading pre-trained model with checkpoint (actual classes):")
    try:
        # Use load_model to load with checkpoint and class mapping
        # num_classes defaults to None, which extracts the actual number of classes from the checkpoint
        model = load_model("efficientnet_animalspeak", device="cpu")
        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print(f"   Number of classes: {output.shape[-1]}")

        # Use class mapping for predictions if available
        if hasattr(model, "label_mapping"):
            print("\n   🏷️  Label mapping available!")
            index_to_label = model.label_mapping["index_to_label"]
            label_to_index = model.label_mapping["label_to_index"]
            print(f"   Total classes in mapping: {len(label_to_index)}")

            # Get top-3 predictions with actual class labels
            probs = torch.softmax(output, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(3, output.shape[-1]), dim=-1)
            print("\n   Top-3 predicted classes:")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0], strict=False)):
                idx_int = idx.item()
                label = index_to_label.get(idx_int, f"Unknown (index {idx_int})")
                print(f"     {i + 1}. {label}: {prob.item():.4f}")

            # Show example of label to index conversion
            print("\n   Example label lookups:")
            example_labels = list(label_to_index.keys())[:3]
            for label in example_labels:
                idx = label_to_index[label]
                print(f"     '{label}' -> index {idx}")
        else:
            print("   ⚠️  No class mapping available (model may not have classifier head)")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   (This is expected if the checkpoint doesn't exist)")

    # Example 2: Create a new model for training
    print("\n🔧 Creating new model for training:")
    try:
        # Use get_model_spec to get the ModelSpec, then get_model to create the model
        model_spec = get_model_spec("sl_beats_animalspeak")
        if model_spec is None:
            print("❌ Model 'sl_beats_animalspeak' not found")
        else:
            model = get_model(model_spec, num_classes=50)
            model = model.cpu()  # Ensure on CPU
            print(f"✅ Created model: {type(model).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Test forward pass
            dummy_input = torch.randn(2, 16000 * 3)  # 3 seconds of audio
            with torch.no_grad():
                output = model(dummy_input, padding_mask=None)
            print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error creating model: {e}")

    # Example 3: Load model with custom parameters
    print("\n🔧 Loading model with custom parameters:")
    try:
        # Create a custom model spec with different parameters
        from representation_learning.configs import AudioConfig, ModelSpec

        custom_spec = ModelSpec(
            name="efficientnet",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="mel_spectrogram",
                n_mels=128,
                target_length_seconds=5,
            ),
            efficientnet_variant="b1",  # Custom variant
        )

        model = get_model(custom_spec, num_classes=25)
        model = model.cpu()  # Ensure on CPU
        print(f"✅ Loaded model with custom parameters: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ Error loading model with custom params: {e}")

    # Example 4: Get model information
    print("\n📊 Model Information:")
    try:
        describe_model("beats_naturelm", verbose=True)

    except Exception as e:
        print(f"❌ Error getting model info: {e}")


if __name__ == "__main__":
    main()
