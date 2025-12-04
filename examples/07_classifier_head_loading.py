"""
Example 7: Classifier Head Loading Behavior

This example demonstrates how load_model handles classifier head weights:
- When num_classes=None: classifier weights are loaded from checkpoint
- When num_classes is explicit: classifier weights are NOT loaded (random init)

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: describe_model("model_name") or get_model_spec("model_name").audio_config.sample_rate
- For full reproducibility, resample using librosa with these exact parameters:

    import librosa
    audio_resampled = librosa.resample(
        audio, orig_sr=original_sr, target_sr=target_sr,
        res_type="kaiser_best", scale=True
    )
"""

import argparse
from pathlib import Path

import torch

from representation_learning import get_model_spec, load_model
from representation_learning.models.get_model import get_model


def main(device: str = "cpu") -> None:
    """Demonstrate classifier head loading behavior."""
    print("Example 7: Classifier Head Loading Behavior")
    print("=" * 60)

    # Ensure checkpoints directory exists
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Part 1: Demonstrating classifier loading with checkpoints
    # =========================================================================
    print("\nPart 1: Checkpoint-based Classifier Loading")
    print("-" * 60)

    # Use the registered sl_beats_animalspeak model
    print("\nCreating BEATs model with classifier...")
    model_spec = get_model_spec("sl_beats_animalspeak")
    model_spec.device = device

    original_num_classes = 15
    model = get_model(model_spec, num_classes=original_num_classes)
    model = model.to(device)

    # Store the original classifier weights
    original_classifier_weight = model.classifier.weight.clone()
    original_classifier_bias = model.classifier.bias.clone()
    print(f"Created model with {original_num_classes} classes")
    print(f"   Classifier weight shape: {original_classifier_weight.shape}")

    # Save checkpoint
    checkpoint_path = checkpoints_dir / "test_beats_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    # Demo 1: Load with num_classes=None (keeps classifier weights from checkpoint)
    print("\nDemo 1: Loading with num_classes=None (default)")
    print("   Behavior: Classifier weights loaded from checkpoint")
    loaded_model_1 = load_model(
        "sl_beats_animalspeak",
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    weights_match = torch.allclose(loaded_model_1.classifier.weight, original_classifier_weight, atol=1e-6)
    bias_match = torch.allclose(loaded_model_1.classifier.bias, original_classifier_bias, atol=1e-6)
    print(f"   Classifier weights match checkpoint: {weights_match and bias_match}")

    # Demo 2: Load with explicit num_classes (random initialization)
    print("\nDemo 2: Loading with explicit num_classes={original_num_classes}")
    print("   Behavior: Classifier weights randomly initialized (not loaded)")
    loaded_model_2 = load_model(
        "sl_beats_animalspeak",
        num_classes=original_num_classes,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    weights_different = not torch.allclose(loaded_model_2.classifier.weight, original_classifier_weight, atol=1e-6)
    print(f"   Classifier weights are new (random): {weights_different}")

    # Demo 3: Load with different num_classes
    print("\nDemo 3: Loading with different num_classes=20")
    print("   Behavior: New classifier created with specified size")
    new_num_classes = 20
    loaded_model_3 = load_model(
        "sl_beats_animalspeak",
        num_classes=new_num_classes,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    print(f"   Classifier output classes: {loaded_model_3.classifier.weight.shape[0]}")

    # =========================================================================
    # Part 2: Self-supervised model (beats_naturelm) use cases
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 2: Self-Supervised Model (beats_naturelm)")
    print("=" * 60)
    print("\nbeats_naturelm is a self-supervised model without a trained classifier.")
    print("This demonstrates the different ways to use such models.\n")

    # Use case 1: Embedding extraction mode (default for models without classifier)
    print("Use Case 1: Embedding Extraction (default behavior)")
    print("-" * 60)
    model = load_model("beats_naturelm", device=device)
    model.eval()

    # Models without a checkpoint classifier load in embedding mode
    has_classifier = hasattr(model, "classifier") and model.classifier is not None
    print(f"   Has classifier: {has_classifier}")
    print(f"   Return features only: {getattr(model, '_return_features_only', 'N/A')}")

    # Test forward pass - returns unpooled frame-level features
    # BEATs expects 16kHz audio
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Output shape: {output.shape} (batch, time_steps, features)")

    # Use case 2: Add a new classification head
    print("\nUse Case 2: Adding a New Classification Head")
    print("-" * 60)
    num_classes = 10
    model = load_model("beats_naturelm", num_classes=num_classes, device=device)
    model.eval()

    print(f"   Classifier weight shape: {model.classifier.weight.shape}")

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Output shape: {output.shape} (batch, num_classes)")

    # Use case 3: Explicit embedding extraction with return_features_only
    print("\nUse Case 3: Explicit Embedding Extraction Mode")
    print("-" * 60)
    model = load_model("beats_naturelm", return_features_only=True, device=device)
    model.eval()

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)
    print(f"   Output shape: {output.shape} (batch, time_steps, features)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. num_classes=None (default):
   - Loads classifier weights from checkpoint if available
   - Otherwise uses embedding extraction mode

2. num_classes=N (explicit):
   - Creates a new randomly initialized classifier
   - Backbone weights still loaded from checkpoint

3. return_features_only=True:
   - Explicitly requests embedding extraction mode
   - Returns unpooled features (batch, time_steps, features)

4. Self-supervised models (like beats_naturelm):
   - No trained classifier exists
   - Default to embedding extraction mode
   - Add classifier via num_classes parameter for fine-tuning
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier Head Loading Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
