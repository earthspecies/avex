"""
Example 1: Basic Model Loading

This example demonstrates the fundamental model loading capabilities:
- Loading pre-trained models with weights
- Creating new models for training
- Using different model types and configurations
"""

import torch

from representation_learning import describe_model, list_models
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Example 1: Basic Model Loading")
    print("=" * 50)

    # List available models
    print("\n📋 Available Models:")
    models = list_models()
    for name, spec in models.items():
        print(f"  - {name}: {spec.name} ({spec.pretrained})")

    # Example 1: Load a pre-trained model with explicit num_classes
    print("\n🔧 Loading pre-trained model with explicit num_classes:")
    try:
        # Use get_model directly for official models (avoid AVES models that download)
        model_spec = models["efficientnet_animalspeak"]
        model = get_model(model_spec, num_classes=10)
        model = model.cpu()  # Ensure on CPU
        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

    # Example 2: Create a new model for training
    print("\n🔧 Creating new model for training:")
    try:
        # Use get_model directly for official models (avoid AVES models that download)
        model_spec = models["sl_beats_animalspeak"]
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
        model_info = describe_model("beats_naturelm")
        print(f"Model: {model_info['_metadata']['name']}")
        print(f"Type: {model_info['_metadata']['model_type']}")
        print(f"Pretrained: {model_info['_metadata']['pretrained']}")
        print(f"Audio Config: {model_info['audio_config']['sample_rate']} Hz")

    except Exception as e:
        print(f"❌ Error getting model info: {e}")


if __name__ == "__main__":
    main()
