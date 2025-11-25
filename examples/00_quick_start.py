"""
Quick Start Example

This example demonstrates the basic functionality that works out of the box:
- Using the original get_model system
- Working with existing model specs
- Basic model loading and testing
"""

import torch

from representation_learning import describe_model, list_models
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Quick Start Example")
    print("=" * 30)

    # List available models (prints table and returns info dict)
    print("\n📋 Available Models:")
    models = list_models()
    # Note: list_models() prints a formatted table above

    if not models:
        print("  No models available")
        return

    # Test with the first available model
    model_name = list(models.keys())[0]
    model_spec = models[model_name]

    print(f"\n🔧 Testing with model: {model_name}")

    try:
        # Use the original get_model system
        model = get_model(model_spec, num_classes=10)
        print(f"✅ Created model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Ensure model is on CPU for this example
        model = model.cpu()
        model.eval()

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio on CPU
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback

        traceback.print_exc()

    # Get model information
    print(f"\n📊 Model Information for {model_name}:")
    try:
        describe_model(model_name, verbose=True)

    except Exception as e:
        print(f"❌ Error getting model info: {e}")

    print("\n🎉 Quick start example completed!")


if __name__ == "__main__":
    main()
