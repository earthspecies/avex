"""
Example 4: Model Registry Management

This example demonstrates:
- Registering custom model configurations
- Managing model specs in the registry
- Working with YAML configuration files
- Registry introspection and management
"""

import tempfile
from pathlib import Path

import yaml

from representation_learning import (
    describe_model,
    get_model_spec,
    list_models,
    register_model,
)
from representation_learning.configs import AudioConfig, ModelSpec


def create_sample_yaml_config() -> dict:
    """Create a sample YAML configuration file.

    Returns:
        dict: A sample configuration dictionary.
    """
    config = {
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
            "optimizer": "adamw",
            "weight_decay": 0.0,
        },
        "dataset_config": {
            "train_datasets": [{"dataset_name": "placeholder"}],
            "val_datasets": [{"dataset_name": "placeholder"}],
        },
        "output_dir": "./runs/placeholder",
        "loss_function": "cross_entropy",
    }
    return config


def main() -> None:
    print("🚀 Example 4: Model Registry Management")
    print("=" * 50)

    # Example 1: Register custom model configurations
    print("\n📋 Registering Custom Model Configurations:")
    try:
        # Create a custom EfficientNet configuration
        custom_efficientnet = ModelSpec(
            name="efficientnet",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=22050,  # Different sample rate
                representation="mel_spectrogram",
                n_mels=256,  # More mel bins
                target_length_seconds=10,  # Longer audio
            ),
            efficientnet_variant="b1",  # Different variant
        )

        register_model("custom_efficientnet", custom_efficientnet)
        print("✅ Registered custom_efficientnet")

        # Create a custom BEATs configuration
        custom_beats = ModelSpec(
            name="beats",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=5),
            use_naturelm=True,
            fine_tuned=True,
        )

        register_model("custom_beats", custom_beats)
        print("✅ Registered custom_beats")

        # List all registered models
        models = list_models()
        print(f"Total registered models: {len(models)}")
        for name in list_models().keys():
            print(f"  - {name}")

    except Exception as e:
        print(f"❌ Error registering models: {e}")

    # Example 2: Work with registered models
    print("\n🔧 Working with Registered Models:")
    try:
        # Get a specific model
        model_spec = get_model_spec("custom_efficientnet")
        if model_spec is not None:
            print("✅ Retrieved custom_efficientnet:")
            print(f"   Model type: {model_spec.name}")
            print(f"   Sample rate: {model_spec.audio_config.sample_rate}")
            print(f"   Variant: {model_spec.efficientnet_variant}")

        # Check if a model is registered
        model_spec = get_model_spec("custom_beats")
        print(f"custom_beats is registered: {model_spec is not None}")

        # Create a model from registered spec using get_model
        from representation_learning.models.get_model import get_model as create_model

        if model_spec is not None:
            model = create_model(model_spec, num_classes=25)
            model = model.cpu()  # Ensure on CPU
            print(f"✅ Created model from registered spec: {type(model).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ Error working with registered models: {e}")

    # Example 3: Model introspection
    print("\n📊 Model Introspection:")
    try:
        # Pretty-print detailed model information
        describe_model("custom_efficientnet", verbose=True)

    except Exception as e:
        print(f"❌ Error in model introspection: {e}")

    # Example 4: Update existing model
    print("\n🔄 Updating Model Configuration:")
    try:
        # Update the custom EfficientNet with different parameters
        updated_efficientnet = ModelSpec(
            name="efficientnet",
            pretrained=True,  # Changed to pretrained
            device="cuda",  # Changed to cuda
            audio_config=AudioConfig(
                sample_rate=16000,  # Changed sample rate
                representation="mel_spectrogram",
                n_mels=128,  # Changed mel bins
                target_length_seconds=5,  # Changed length
            ),
            efficientnet_variant="b0",  # Changed variant
        )

        register_model("custom_efficientnet", updated_efficientnet)
        print("✅ Updated custom_efficientnet configuration")

        # Verify the update
        model_spec = get_model_spec("custom_efficientnet")
        if model_spec is not None:
            print(f"   New sample rate: {model_spec.audio_config.sample_rate}")
            print(f"   New variant: {model_spec.efficientnet_variant}")
            print(f"   New pretrained: {model_spec.pretrained}")

    except Exception as e:
        print(f"❌ Error updating model: {e}")

    # Example 5: Load from YAML configuration file
    print("\n📄 Loading from YAML Configuration:")
    try:
        # Create a temporary YAML file
        config = create_sample_yaml_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config, f)
            yaml_path = f.name

        print(f"Created YAML config at: {yaml_path}")

        # Load model from YAML file using get_model
        from representation_learning.models.get_model import get_model as create_model
        from representation_learning.models.utils.registry import (
            load_model_spec_from_yaml,
        )

        model_spec = load_model_spec_from_yaml(yaml_path)
        model = create_model(model_spec, num_classes=30)
        model = model.cpu()  # Ensure on CPU
        print(f"✅ Created model from YAML: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # The YAML model should be auto-registered
        yaml_model_name = Path(yaml_path).stem
        model_spec = get_model_spec(yaml_model_name)
        print(f"YAML model auto-registered: {model_spec is not None}")

        # Clean up
        Path(yaml_path).unlink()

    except Exception as e:
        print(f"❌ Error loading from YAML: {e}")

    # Example 6: Note on model registration
    print("\n📝 Note on Model Registration:")
    print("   Custom models remain registered for the session.")
    print("   To use different configurations, register with unique names.")
    print("   Example: register_model('custom_efficientnet_v2', new_spec)")

    # Check registration status
    models = list_models()
    is_reg1 = "custom_efficientnet" in models
    is_reg2 = "custom_beats" in models
    print(f"   custom_efficientnet still registered: {is_reg1}")
    print(f"   custom_beats still registered: {is_reg2}")

    # Models remain registered and can be used
    try:
        model = create_model("custom_efficientnet", num_classes=10, device="cpu")
        print(f"✅ Model creation succeeded: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")

    # Example 7: Registry statistics
    print("\n📈 Registry Statistics:")
    try:
        models = list_models()
        model_names = list(models.keys())

        print(f"Total registered models: {len(models)}")
        print(f"Model names: {model_names}")

        # Count by model type
        model_types = {}
        for _name, spec in models.items():
            model_type = spec.name
            model_types[model_type] = model_types.get(model_type, 0) + 1

        print("Models by type:")
        for model_type, count in model_types.items():
            print(f"  - {model_type}: {count}")

    except Exception as e:
        print(f"❌ Error getting registry statistics: {e}")


if __name__ == "__main__":
    main()
