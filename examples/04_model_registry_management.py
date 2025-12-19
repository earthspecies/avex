"""
Example 4: Model Registry Management

This example demonstrates:
- Registering custom model configurations
- Managing model specs in the registry
- Working with YAML configuration files
- Registry introspection

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
import tempfile
from pathlib import Path

import yaml

from representation_learning import (
    describe_model,
    get_model_spec,
    list_models,
    register_model,
)
from representation_learning.configs import AudioConfig, ModelSpec, ProbeConfig
from representation_learning.models.probes.utils import build_probe_from_config_online
from representation_learning.models.utils.factory import build_model_from_spec
from representation_learning.models.utils.registry import load_model_spec_from_yaml


def create_sample_yaml_config() -> dict:
    """Create a sample YAML configuration.

    Returns
    -------
    dict
        Sample YAML configuration dictionary.
    """
    return {
        "model_spec": {
            "name": "efficientnet",
            "pretrained": False,
            "device": "cpu",
            "audio_config": {
                "sample_rate": 16000,
                "representation": "mel_spectrogram",
                "n_mels": 128,
                "target_length_seconds": 5,
            },
            "efficientnet_variant": "b1",
        },
        "training_params": {
            "train_epochs": 1,
            "lr": 0.001,
            "batch_size": 1,
        },
    }


def main(device: str = "cpu") -> None:
    """Demonstrate model registry management."""
    print("Example 4: Model Registry Management")
    print("=" * 50)

    # =========================================================================
    # Part 1: Register custom model configurations
    # =========================================================================
    print("\nPart 1: Register Custom Configurations")
    print("-" * 50)

    # Register custom EfficientNet
    custom_efficientnet = ModelSpec(
        name="efficientnet",
        pretrained=False,
        device=device,
        audio_config=AudioConfig(
            sample_rate=22050,
            representation="mel_spectrogram",
            n_mels=256,
            target_length_seconds=10,
        ),
        efficientnet_variant="b1",
    )
    register_model("custom_efficientnet", custom_efficientnet)
    print("Registered: custom_efficientnet")
    print(f"   Sample rate: {custom_efficientnet.audio_config.sample_rate}")
    print(f"   Variant: {custom_efficientnet.efficientnet_variant}")

    # Register custom BEATs
    custom_beats = ModelSpec(
        name="beats",
        pretrained=False,
        device=device,
        audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=5),
        use_naturelm=True,
        fine_tuned=True,
    )
    register_model("custom_beats", custom_beats)
    print("\nRegistered: custom_beats")
    print(f"   Use NatureLM: {custom_beats.use_naturelm}")

    # Show total models
    models = list_models()
    print(f"\nTotal registered models: {len(models)}")

    # =========================================================================
    # Part 2: Work with registered models
    # =========================================================================
    print("\nPart 2: Work with Registered Models")
    print("-" * 50)

    # Retrieve custom model spec
    model_spec = get_model_spec("custom_efficientnet")
    print("Retrieved custom_efficientnet:")
    print(f"   Model type: {model_spec.name}")
    print(f"   Sample rate: {model_spec.audio_config.sample_rate}")

    # Create backbone model from registered spec
    model_spec = get_model_spec("custom_beats")
    backbone = build_model_from_spec(model_spec, device=device)

    # Attach a simple linear probe for classification
    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["last_layer"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    model = build_probe_from_config_online(
        probe_config=probe_config,
        base_model=backbone,
        num_classes=25,
        device=device,
    ).cpu()

    print(f"\nCreated model with linear probe from custom_beats: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # Part 3: Model introspection
    # =========================================================================
    print("\nPart 3: Model Introspection")
    print("-" * 50)
    describe_model("custom_efficientnet", verbose=True)

    # =========================================================================
    # Part 4: Update existing model
    # =========================================================================
    print("\nPart 4: Update Model Configuration")
    print("-" * 50)

    updated_efficientnet = ModelSpec(
        name="efficientnet",
        pretrained=True,
        device=device,
        audio_config=AudioConfig(
            sample_rate=16000,
            representation="mel_spectrogram",
            n_mels=128,
            target_length_seconds=5,
        ),
        efficientnet_variant="b0",
    )
    register_model("custom_efficientnet", updated_efficientnet)
    print("Updated custom_efficientnet configuration:")

    model_spec = get_model_spec("custom_efficientnet")
    print(f"   New sample rate: {model_spec.audio_config.sample_rate}")
    print(f"   New variant: {model_spec.efficientnet_variant}")
    print(f"   New pretrained: {model_spec.pretrained}")

    # =========================================================================
    # Part 5: Load from YAML configuration
    # =========================================================================
    print("\nPart 5: Load from YAML Configuration")
    print("-" * 50)

    config = create_sample_yaml_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(config, f)
        yaml_path = f.name

    print(f"Created YAML config: {yaml_path}")

    model_spec = load_model_spec_from_yaml(yaml_path)
    backbone = build_model_from_spec(model_spec, device=device)

    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["last_layer"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    model = build_probe_from_config_online(
        probe_config=probe_config,
        base_model=backbone,
        num_classes=30,
        device=device,
    ).cpu()

    print(f"Created model with linear probe from YAML: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Clean up
    Path(yaml_path).unlink()

    # =========================================================================
    # Part 6: Registry statistics
    # =========================================================================
    print("\nPart 6: Registry Statistics")
    print("-" * 50)

    models = list_models()
    print(f"Total registered models: {len(models)}")

    # Count by model type
    model_types: dict[str, int] = {}
    for info in models.values():
        model_type = info.get("model_type", "unknown")
        model_types[model_type] = model_types.get(model_type, 0) + 1

    print("\nModels by type:")
    for model_type, count in sorted(model_types.items()):
        print(f"  - {model_type}: {count}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("""
- register_model(name, spec): Add ModelSpec to registry
- get_model_spec(name): Retrieve spec from registry
- describe_model(name): Display detailed model info
- load_model_spec_from_yaml(path): Load spec from YAML file
- Registry persists for the session
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Registry Management Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
