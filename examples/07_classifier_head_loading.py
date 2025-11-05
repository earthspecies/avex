"""
Example 7: Classifier Head Loading Behavior

This example demonstrates how load_model handles classifier head weights:
- When num_classes=None: classifier weights are loaded from checkpoint
- When num_classes is explicit: classifier weights are NOT loaded (random init)
"""

import torch

from representation_learning import (
    get_model as get_model_spec,
)
from representation_learning import (
    load_model,
    register_model_class,
    unregister_model_class,
)
from representation_learning.models.beats_model import Model as BeatsModel
from representation_learning.models.get_model import get_model


def main() -> None:
    print("🚀 Example 7: Classifier Head Loading Behavior")
    print("=" * 60)

    # Register the BEATs model class so build_model_from_spec can find it
    original_name = getattr(BeatsModel, "name", None)
    BeatsModel.name = "beats"  # type: ignore[attr-defined]
    try:
        register_model_class(BeatsModel)
        print("✅ Registered BEATs model class")

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
            print("\n📋 Test 1: Loading with num_classes=None")
            print("   Expected: Classifier weights should match checkpoint")
            loaded_model_1 = load_model(
                "sl_beats_animalspeak",
                num_classes=None,  # Should extract from checkpoint
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
                print(
                    "   ❌ FAIL: Classifier weights match checkpoint (should be random)"
                )

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
                print(
                    f"      Classifier weight shape: "
                    f"{loaded_model_3.classifier.weight.shape}"
                )
            else:
                print(
                    f"   ❌ FAIL: Expected {new_num_classes} classes, "
                    f"got {loaded_model_3.classifier.weight.shape[0]}"
                )

        finally:
            # Note: Checkpoint saved to checkpoints/ directory for future use
            print(f"\n📁 Checkpoint saved to: {checkpoint_path}")
            print("   (Checkpoint preserved in checkpoints/ directory)")
            # Optionally clean up:
            # if checkpoint_path.exists():
            #     checkpoint_path.unlink()
            #     print("🧹 Cleaned up checkpoint file")
    finally:
        # Cleanup: unregister and restore original name
        unregister_model_class("beats")
        if original_name is not None:
            BeatsModel.name = original_name  # type: ignore[attr-defined]
        elif hasattr(BeatsModel, "name"):
            delattr(BeatsModel, "name")
        print("🧹 Cleaned up registered model class")

    print("\n✅ Example completed successfully")


if __name__ == "__main__":
    main()
