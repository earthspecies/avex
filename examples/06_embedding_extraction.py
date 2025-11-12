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

from representation_learning import load_model


def main() -> None:
    print("🚀 Example 8: Embedding Extraction Mode")
    print("=" * 50)

    # Example 1: Load BEATs NatureLM for embedding extraction
    # This loads the BEATs model without a classifier head
    print("\n📋 Example 1: Loading BEATs NatureLM for Embedding Extraction")
    print("-" * 50)
    try:
        # Load without num_classes - automatically uses return_features_only=True
        print("Loading BEATs NatureLM model without num_classes (embedding extraction mode)...")
        model = load_model("beats_naturelm", num_classes=None, device="cpu")
        model.eval()

        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Return features only: {getattr(model, '_return_features_only', 'N/A')}")

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

    # Example 2: Load EAT-HF for embedding extraction
    print("\n📋 Example 2: Loading EAT-HF for Embedding Extraction")
    print("-" * 50)
    try:
        # Load without num_classes - automatically uses return_features_only=True
        print("Loading EAT-HF model without num_classes (embedding extraction mode)...")
        model = load_model("sl_eat_animalspeak_ssl_all", num_classes=None, device="cpu")
        model.eval()

        print(f"✅ Loaded model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Return features only: {getattr(model, '_return_features_only', 'N/A')}")

        # Test forward pass
        dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)
        print(f"   Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
        print("   ✅ Model returns embeddings (not classification logits)")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()

    # Example 3: Compare embedding vs classification mode using the same model
    print("\n📋 Example 3: Comparison - Embedding vs Classification Mode (same model)")
    print("-" * 50)
    try:
        # Load in embedding mode (return_features_only=True strips classifier from checkpoint)
        print("Loading sl_beats_all in embedding extraction mode (return_features_only=True)...")
        print("   (Same checkpoint used, but classifier head is stripped)")
        embedding_model = load_model("sl_beats_all", num_classes=None, return_features_only=True, device="cpu")
        embedding_model.eval()

        # Load in classification mode (with checkpoint, extracts actual classes and class mapping)
        print("Loading sl_beats_all in classification mode (with checkpoint and class mapping)...")
        print("   (Same checkpoint used, classifier head is preserved)")
        classification_model = load_model("sl_beats_all", num_classes=None, device="cpu")
        classification_model.eval()

        # Check if class mapping is available
        if hasattr(classification_model, "class_mapping"):
            index_to_label = classification_model.class_mapping["index_to_label"]
            label_to_index = classification_model.class_mapping["label_to_index"]
            print(f"   ✅ Class mapping loaded: {len(label_to_index)} classes")

        # Test both with same input
        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            embedding_output = embedding_model(dummy_input, padding_mask=None)
            classification_output = classification_model(dummy_input, padding_mask=None)

        print(f"\n   Embedding mode output shape: {embedding_output.shape}")
        print(f"   Classification mode output shape: {classification_output.shape}")

        # Show example prediction with class labels
        if hasattr(classification_model, "class_mapping"):
            probs = torch.softmax(classification_output, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(3, classification_output.shape[-1]), dim=-1)
            print("\n   Top-3 predicted classes (with labels):")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0], strict=False)):
                idx_int = idx.item()
                label = index_to_label.get(idx_int, f"Unknown (index {idx_int})")
                print(f"     {i + 1}. {label}: {prob.item():.4f}")

        print("   ✅ Both modes work correctly with the same model!")

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
