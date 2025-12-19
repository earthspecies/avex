"""
Example 7: Classifier Head and Probe Behavior

This example demonstrates how load_model and probe heads interact:
- How classifier weights are preserved when loading from checkpoints
- How to use return_features_only for embedding extraction
- How to attach a new linear probe head on top of a backbone

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

from representation_learning import load_model
from representation_learning.configs import ProbeConfig
from representation_learning.models.probes.utils import build_probe_from_config_online


def main(device: str = "cpu") -> None:
    """Demonstrate classifier head and probe behavior.

    Args:
        device: Device to use for model and data.

    Raises:
        ValueError: If model does not have a classifier when expected.
    """
    print("Example 7: Classifier Head and Probe Behavior")
    print("=" * 60)

    # Ensure checkpoints directory exists
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Part 1: Demonstrating classifier loading with checkpoints
    # =========================================================================
    print("\nPart 1: Checkpoint-based classifier loading")
    print("-" * 60)

    # Use the registered sl_beats_animalspeak model with its checkpoint classifier
    print("\nLoading BEATs model with classifier from checkpoint ...")
    model = load_model("sl_beats_animalspeak", device=device)
    model = model.to(device)

    # Check if model has a classifier (it should if loaded from checkpoint with classifier weights)
    if not hasattr(model, "classifier") or model.classifier is None:
        raise ValueError(
            "Model does not have a classifier. This might happen if the checkpoint "
            "doesn't contain classifier weights or if the model was loaded in "
            "return_features_only mode."
        )

    # Store the original classifier weights from the checkpoint
    original_classifier_weight = model.classifier.weight.clone()
    original_classifier_bias = model.classifier.bias.clone()
    original_num_classes = original_classifier_weight.shape[0]
    print(f"Loaded model with {original_num_classes} classes")
    print(f"   Classifier weight shape: {original_classifier_weight.shape}")

    # Save checkpoint
    checkpoint_path = checkpoints_dir / "test_beats_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    # Demo 1: Load from explicit checkpoint (keeps classifier weights)
    print("\nDemo 1: Loading from explicit checkpoint")
    print("   Behavior: Classifier weights loaded from checkpoint")
    loaded_model_1 = load_model(
        "sl_beats_animalspeak",
        checkpoint_path=str(checkpoint_path),
        device=device,
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
    print(f"   Classifier weights match checkpoint: {weights_match and bias_match}")

    # =========================================================================
    # Part 2: Self-supervised model (beats_naturelm) use cases
    # =========================================================================
    print("\n" + "=" * 60)
    print("Part 2: Self-supervised model (beats_naturelm)")
    print("=" * 60)
    print("\nbeats_naturelm is a self-supervised model without a trained classifier.")
    print("This demonstrates different ways to use such models.\n")

    # Use case 1: Embedding extraction mode (default for models without classifier)
    print("Use case 1: Embedding extraction (default behavior)")
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

    # Use case 2: Add a new classification head via linear probe
    print("\nUse case 2: Adding a new classification head with linear probe")
    print("-" * 60)
    num_classes = 10

    backbone = load_model("beats_naturelm", device=device, return_features_only=True)
    backbone.eval()

    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["backbone"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    probe = build_probe_from_config_online(
        probe_config=probe_config,
        base_model=backbone,
        num_classes=num_classes,
        device=device,
    )
    probe.eval()

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        logits = probe(dummy_input)
    print(f"   Probe output shape: {logits.shape} (batch, num_classes)")

    # Use case 3: Explicit embedding extraction with return_features_only
    print("\nUse case 3: Explicit embedding extraction mode")
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
    print("Key takeaways")
    print("=" * 60)
    print("""
1. Supervised models with checkpoint classifiers:
   - load_model keeps the classifier weights from the checkpoint

2. Self-supervised models (like beats_naturelm):
   - No trained classifier exists, so they default to embedding extraction mode

3. return_features_only=True:
   - Explicitly requests embedding extraction mode
   - Returns unpooled features (batch, time_steps, features)

4. Probe heads:
   - You can attach a simple linear probe via build_probe_from_config
   - Backbones stay reusable across tasks and heads
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
