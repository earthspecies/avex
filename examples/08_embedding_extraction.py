"""
Example 8: Embedding Extraction Mode

This example demonstrates loading models without a classification head
for embedding extraction, which is useful for representation learning tasks.

Key use cases:
- Loading pre-trained models for feature extraction
- Building models without classifiers for embedding analysis
- Using models with return_features_only=True
"""

import torch

from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Example 8: Embedding Extraction Mode")
    print("=" * 50)

    # Example 1: Load BEATs with pretrained=True and use_naturelm=False (default)
    # This loads the original BEATs model without a checkpoint
    print("\n📋 Example 1: Loading pretrainedBEATs for Embedding Extraction")
    print("-" * 50)
    try:
        beats_spec = ModelSpec(
            name="beats",
            pretrained=True,  # Use pretrained weights
            device="cpu",
            use_naturelm=False,  # Default: use standard BEATs (not NatureLM)
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="raw",
                target_length_seconds=10,
            ),
        )

        # Load without num_classes - automatically uses return_features_only=True
        print("Loading BEATs model without num_classes (embedding extraction mode)...")
        # Create model directly for embedding extraction (return_features_only=True)
        from representation_learning.models.beats_model import Model as BEATsModel

        model = BEATsModel(
            num_classes=1,  # Dummy value (not used when return_features_only=True)
            pretrained=beats_spec.pretrained,
            device="cpu",
            audio_config=beats_spec.audio_config,
            return_features_only=True,
            use_naturelm=beats_spec.use_naturelm or False,
        )
        model = model.cpu()
        model.eval()

        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"   Return features only: {getattr(model, '_return_features_only', 'N/A')}"
        )

        # Test forward pass - should return embeddings, not logits
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print("   ✅ Model returns embeddings (not classification logits)")

    except Exception as e:
        print(f"❌ Error loading BEATs model: {e}")
        import traceback

        traceback.print_exc()

    # Example 2: Load EAT-HF with pretrained=True for embedding extraction
    # Using configuration from configs/run_configs/pretrained/eat_base.yml
    print("\n📋 Example 2: Loading pretrained EAT-HF for Embedding Extraction")
    print("-" * 50)
    try:
        eat_spec = ModelSpec(
            name="eat_hf",
            pretrained=True,  # Use pretrained weights
            device="cpu",
            model_id="worstchan/EAT-base_epoch30_pretrain",  # From eat_base.yml
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="raw",
                normalize=False,
                target_length_seconds=10,
                window_selection="random",
            ),
        )

        # Load without num_classes - automatically uses return_features_only=True
        print("Loading EAT-HF model without num_classes (embedding extraction mode)...")
        # Create model directly for embedding extraction (return_features_only=True)
        from representation_learning.models.eat_hf import Model as EATHFModel

        model = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",  # From eat_base.yml
            num_classes=1,  # Dummy value (not used when return_features_only=True)
            device="cpu",
            audio_config=eat_spec.audio_config,
            return_features_only=True,
            # Using default normalization parameters (as in eat_base.yml)
            norm_mean=-4.268,
            norm_std=4.569,
        )
        model = model.cpu()
        model.eval()

        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"   Return features only: {getattr(model, 'return_features_only', 'N/A')}"
        )

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print("   ✅ Model returns embeddings (not classification logits)")

    except Exception as e:
        print(f"❌ Error loading EAT-HF model: {e}")
        import traceback

        traceback.print_exc()

    # Example 3: Compare with classification mode
    print("\n📋 Example 3: Comparison - Embedding vs Classification Mode")
    print("-" * 50)
    try:
        spec = ModelSpec(
            name="efficientnet",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="mel_spectrogram",
                target_length_seconds=5,
            ),
        )

        # Load in embedding mode (no num_classes)
        print("Loading in embedding extraction mode (return_features_only=True)...")
        from representation_learning.models.efficientnet import Model as EfficientNet

        embedding_model = EfficientNet(
            num_classes=1,  # Dummy value (not used when return_features_only=True)
            pretrained=spec.pretrained,
            device="cpu",
            audio_config=spec.audio_config,
            return_features_only=True,
            efficientnet_variant=getattr(spec, "efficientnet_variant", "b0"),
        )
        embedding_model = embedding_model.cpu()
        embedding_model.eval()

        # Load in classification mode (with num_classes)
        print("Loading in classification mode (num_classes=10)...")
        classification_model = get_model(spec, num_classes=10)
        classification_model = classification_model.cpu()
        classification_model.eval()

        # Test both with same input
        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            embedding_output = embedding_model(dummy_input, padding_mask=None)
            classification_output = classification_model(dummy_input, padding_mask=None)

        print(f"\n   Embedding mode output shape: {embedding_output.shape}")
        print(f"   Classification mode output shape: {classification_output.shape}")
        print("   ✅ Both modes work correctly")

    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback

        traceback.print_exc()

    print("\n🎉 Embedding extraction example completed!")
    print("\n💡 Key Takeaways:")
    print("   - Models can be loaded without num_classes for embedding extraction")
    print("   - Automatically uses return_features_only=True when supported")
    print("   - Useful for representation learning and feature analysis")
    print("   - Works with pretrained=True for models like BEATs and EAT-HF")


if __name__ == "__main__":
    main()
