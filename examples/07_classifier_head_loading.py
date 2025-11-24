"""
Example 7: Classifier Head Loading Behavior

This example demonstrates how load_model handles classifier head weights:
- When num_classes=None: classifier weights are loaded from checkpoint
- When num_classes is explicit: classifier weights are NOT loaded (random init)
"""

import torch

from representation_learning import (
    get_model_spec,
    load_model,
)
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Example 7: Classifier Head Loading Behavior")
    print("=" * 60)

    # Note: BEATs model class is now auto-registered at startup
    # Use checkpoints directory for test checkpoint
    from pathlib import Path

    # Ensure checkpoints directory exists
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    checkpoint_dir = checkpoints_dir

    # Use the registered sl_beats_animalspeak model
    print("\n📦 Creating BEATs model with classifier...")
    model_spec = get_model_spec("sl_beats_animalspeak")
    if model_spec is None:
        print("❌ Error: sl_beats_animalspeak model not found in registry")
        return

    # Force CPU device for this example
    model_spec.device = "cpu"

    original_num_classes = 15
    model = get_model(model_spec, num_classes=original_num_classes)
    model = model.to("cpu")

    # Store the original classifier weights
    original_classifier_weight = model.classifier.weight.clone()
    original_classifier_bias = model.classifier.bias.clone()
    print(f"✅ Created model with {original_num_classes} classes")
    print(f"   Classifier weight shape: {original_classifier_weight.shape}")
    print(f"   Classifier bias shape: {original_classifier_bias.shape}")

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "test_beats_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Saved checkpoint to: {checkpoint_path}")
    print("   (Checkpoint saved to checkpoints/ directory)")

    try:
        # Test 1: Load with num_classes=None (should keep classifier weights)
        print("\n📋 Test 1: Loading with num_classes=None (default)")
        print("   Expected: Classifier weights should match checkpoint")
        loaded_model_1 = load_model(
            "sl_beats_animalspeak",
            checkpoint_path=str(checkpoint_path),  # Use saved checkpoint
            device="cpu",
        )

        weights_match = torch.allclose(
            loaded_model_1.classifier.weight,
            original_classifier_weight,
            atol=1e-6,
        )
        bias_match = torch.allclose(
            loaded_model_1.classifier.bias,
            original_classifier_bias,
            atol=1e-6,
        )

        if weights_match and bias_match:
            print("   ✅ SUCCESS: Classifier weights match checkpoint")
        else:
            print("   ❌ FAIL: Classifier weights do NOT match checkpoint")
            print(f"      Weights match: {weights_match}")
            print(f"      Bias match: {bias_match}")

        # Test 2: Load with explicit num_classes (should NOT load classifier)
        print("\n📋 Test 2: Loading with explicit num_classes (same as checkpoint)")
        print("   Expected: Classifier weights should be randomly initialized")
        loaded_model_2 = load_model(
            "sl_beats_animalspeak",
            num_classes=original_num_classes,  # Explicit, matches
            checkpoint_path=str(checkpoint_path),  # Use saved checkpoint
            device="cpu",
        )

        weights_different = not torch.allclose(
            loaded_model_2.classifier.weight,
            original_classifier_weight,
            atol=1e-6,
        )
        bias_different = not torch.allclose(
            loaded_model_2.classifier.bias,
            original_classifier_bias,
            atol=1e-6,
        )

        if weights_different or bias_different:
            print("   ✅ SUCCESS: Classifier weights are randomly initialized")
        else:
            print("   ❌ FAIL: Classifier weights match checkpoint (should be random)")

        # Test 3: Load with different num_classes
        print("\n📋 Test 3: Loading with different num_classes")
        print("   Expected: New classifier with different shape")
        new_num_classes = 20
        loaded_model_3 = load_model(
            "sl_beats_animalspeak",
            num_classes=new_num_classes,  # Different from checkpoint
            checkpoint_path=str(checkpoint_path),  # Use saved checkpoint
            device="cpu",
        )

        if loaded_model_3.classifier.weight.shape[0] == new_num_classes:
            print(f"   ✅ SUCCESS: Classifier has {new_num_classes} classes")
            print(f"      Classifier weight shape: {loaded_model_3.classifier.weight.shape}")
        else:
            print(f"   ❌ FAIL: Expected {new_num_classes} classes, got {loaded_model_3.classifier.weight.shape[0]}")

    finally:
        # Note: Checkpoint saved to checkpoints/ directory for future use
        print(f"\n📁 Checkpoint saved to: {checkpoint_path}")
        print("   (Checkpoint preserved in checkpoints/ directory)")
        # Optionally clean up:
        # if checkpoint_path.exists():
        #     checkpoint_path.unlink()
        #     print("🧹 Cleaned up checkpoint file")

    print("\n✅ Example completed successfully")

    # Example: beats_naturelm (self-supervised model without checkpoint/classifier)
    print("\n" + "=" * 60)
    print("📋 Example: beats_naturelm - Self-Supervised Model Use Cases")
    print("=" * 60)
    print("\n   beats_naturelm is a self-supervised model (no checkpoint, no classifier)")
    print("   This demonstrates three different use cases:\n")

    # Use case 1: Try to load with original classification head (should not have one)
    print("📋 Use Case 1: Trying to load with original classification head")
    print("   Expected: Cannot load original classifier (none exists - model loads in embedding mode)")
    print("-" * 60)
    try:
        # beats_naturelm has no checkpoint and no classifier, so loading without num_classes
        # will automatically use return_features_only=True (embedding extraction mode)
        model = load_model("beats_naturelm", device="cpu")
        model.eval()

        # Check if classifier exists and is not None (BEATs sets classifier=None when return_features_only=True)
        has_classifier = hasattr(model, "classifier") and model.classifier is not None
        if has_classifier:
            print("   ❌ UNEXPECTED: Model has a classifier (should not exist)")
            print("   (beats_naturelm has no checkpoint, so no original classifier exists)")
        else:
            print("   ✅ EXPECTED: Model has no classifier (no checkpoint/classifier exists)")
            print("   ✅ Model automatically loaded in embedding extraction mode")
            print(f"      Return features only: {getattr(model, '_return_features_only', 'N/A')}")

            # Test forward pass - should return embeddings, not logits
            dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
            with torch.no_grad():
                output = model(dummy_input, padding_mask=None)
            print(f"      Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
            print("      ✅ Model returns embeddings (not classification logits)")
            print("   💡 Note: You cannot load the 'original' classification head")
            print("      because beats_naturelm is self-supervised (no classifier trained)")

    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Use case 2: Add a new classification head through num_classes
    print("\n📋 Use Case 2: Adding a new classification head through num_classes")
    print("   Expected: Should create a new randomly initialized classifier")
    print("-" * 60)
    try:
        num_classes = 10
        model = load_model("beats_naturelm", num_classes=num_classes, device="cpu")
        model.eval()

        if hasattr(model, "classifier"):
            print(f"   ✅ SUCCESS: Model has a classifier with {num_classes} classes")
            print(f"      Classifier weight shape: {model.classifier.weight.shape}")
            print(f"      Classifier bias shape: {model.classifier.bias.shape}")

            # Test forward pass
            dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
            with torch.no_grad():
                output = model(dummy_input, padding_mask=None)
            print(f"      Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
            print("      ✅ Model outputs classification logits")
        else:
            print("   ❌ FAIL: Model does not have a classifier")
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Use case 3: Use for feature/embedding extraction
    print("\n📋 Use Case 3: Using for feature/embedding extraction")
    print("   Expected: Should return embeddings (no classifier)")
    print("-" * 60)
    try:
        # Load without num_classes - should automatically use return_features_only=True
        model = load_model("beats_naturelm", device="cpu")
        model.eval()

        # Check if classifier exists and is not None (BEATs sets classifier=None when return_features_only=True)
        has_classifier = hasattr(model, "classifier") and model.classifier is not None
        if has_classifier:
            print("   ❌ UNEXPECTED: Model has a classifier (should be in embedding mode)")
        else:
            print("   ✅ SUCCESS: Model is in embedding extraction mode")
            print(f"      Return features only: {getattr(model, '_return_features_only', 'N/A')}")

            # Test forward pass - should return embeddings
            dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
            with torch.no_grad():
                output = model(dummy_input, padding_mask=None)
            print(f"      Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
            print("      ✅ Model returns embeddings (not classification logits)")

    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n💡 Key Takeaways for beats_naturelm:")
    print("   - Cannot load with original classification head (no checkpoint exists)")
    print("   - Can add a new classification head by specifying num_classes")
    print("   - Automatically uses embedding extraction mode when num_classes=None")
    print("   - Useful for transfer learning and representation learning tasks")


if __name__ == "__main__":
    main()
