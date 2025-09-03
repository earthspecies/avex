"""Test LSTM probe embedding extraction to debug dimension issues."""

import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel
from representation_learning.models.probes.lstm_probe import LSTMProbe


class TestLSTMProbeEmbeddings:
    """Test LSTM probe embedding extraction and dimension handling."""

    def test_efficientnet_embedding_extraction_debug(self) -> None:
        """Test EfficientNet embedding extraction with the same config as the
        failing case."""
        # Create EfficientNet model with same config as
        # sl_efficientnet_animalspeak_alllayers.yml
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            efficientnet_variant="b0",
        )

        # Create dummy audio input (batch_size=1, time=16000) - same as error case
        x = torch.randn(1, 16000)

        print(f"Input shape: {x.shape}")
        print("Target layer: model.features.8")

        # Test the exact configuration from the failing case
        try:
            print("\nTesting embedding extraction with aggregation='none'...")
            # Register hooks for the target layer
            model.register_hooks_for_layers(["model.features.8"])
            embeddings = model.extract_embeddings(
                x=x,
                aggregation="none",
            )

            print(f"SUCCESS! Embeddings type: {type(embeddings)}")

            if isinstance(embeddings, list):
                print(f"Got list with {len(embeddings)} items")
                for i, emb in enumerate(embeddings):
                    print(
                        f"  Item {i}: type={type(emb)}, "
                        f"shape={emb.shape if hasattr(emb, 'shape') else 'no shape'}"
                    )

                # Check if any item in the list is 3D
                has_3d = any(
                    hasattr(emb, "shape") and emb.dim() == 3 for emb in embeddings
                )
                if has_3d:
                    print(
                        "✓ List contains 3D embeddings - this should work with "
                        "LSTM probe!"
                    )
                else:
                    print("✗ List contains no 3D embeddings - LSTM probe will fail")

            elif isinstance(embeddings, torch.Tensor):
                print(f"SUCCESS! Embeddings shape: {embeddings.shape}")
                print(f"Embeddings dimensions: {embeddings.dim()}")
                print(f"Embeddings type: {type(embeddings)}")

                # Check if this is what LSTM probe expects
                if embeddings.dim() == 3:
                    batch_size, time_steps, feature_dim = embeddings.shape
                    print(
                        f"✓ 3D embeddings: batch_size={batch_size}, "
                        f"time_steps={time_steps}, feature_dim={feature_dim}"
                    )
                    print("✓ This should work with LSTM probe!")
                else:
                    print(f"✗ Expected 3D embeddings for LSTM, got {embeddings.dim()}D")

            else:
                print(f"✗ Expected tensor or list, got {type(embeddings)}")

        except Exception as e:
            print(f"ERROR in embedding extraction: {e}")
            print(f"Error type: {type(e)}")
            raise

        # Now test what happens with the default parameters (what was happening before)
        try:
            print(
                "\nTesting embedding extraction with default parameters "
                "(aggregation='mean')..."
            )
            embeddings_default = model.extract_embeddings(
                x=x,
                # aggregation="mean" (default)
            )

            print(f"Default params result - shape: {embeddings_default.shape}")
            print(f"Default params result - dimensions: {embeddings_default.dim()}")

            if isinstance(embeddings_default, torch.Tensor):
                if embeddings_default.dim() == 2:
                    print(
                        "✗ Default params give 2D embeddings - this is why "
                        "LSTM probe failed!"
                    )
                elif embeddings_default.dim() == 3:
                    print("✓ Default params give 3D embeddings - unexpected!")
                else:
                    print(
                        f"? Default params give {embeddings_default.dim()}D embeddings"
                    )

        except Exception as e:
            print(f"ERROR with default params: {e}")

    def test_lstm_probe_creation_with_embeddings(self) -> None:
        """Test creating LSTM probe with the embeddings we just extracted."""
        # Create EfficientNet model
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            efficientnet_variant="b0",
        )

        # Create dummy audio input
        x = torch.randn(1, 16000)

        # Register hooks for the target layer
        model.register_hooks_for_layers(["model.features.8"])

        # Extract embeddings with LSTM-compatible settings
        embeddings = model.extract_embeddings(
            x=x,
            aggregation="none",
        )

        print(f"\nExtracted embeddings for LSTM probe: {type(embeddings)}")

        # Handle the case where extract_embeddings returns a list
        if isinstance(embeddings, list):
            print(f"Got list with {len(embeddings)} items")
            if len(embeddings) > 0:
                first_embedding = embeddings[0]
                print(f"First embedding shape: {first_embedding.shape}")
                print(f"First embedding dimensions: {first_embedding.dim()}")

                if first_embedding.dim() == 3:
                    print("✓ First embedding is 3D - this should work with LSTM probe!")
                else:
                    print(f"✗ First embedding is not 3D: {first_embedding.dim()}D")
                    return
            else:
                print("✗ Empty list returned")
                return
        elif isinstance(embeddings, torch.Tensor):
            print(f"Got tensor directly: {embeddings.shape}")
        else:
            print(f"✗ Unexpected return type: {type(embeddings)}")
            return

        # Try to create LSTM probe
        try:
            probe = LSTMProbe(
                base_model=model,
                layers=["model.features.8"],
                num_classes=10,
                device="cpu",
                feature_mode=False,
                aggregation="none",  # Use 'none' to get 3D embeddings for LSTM
                lstm_hidden_size=256,
                num_layers=2,
                bidirectional=True,
                max_sequence_length=1000,
                use_positional_encoding=False,
                target_length=16000,  # Provide target_length to fix initialization
            )
            print("✓ LSTM probe created successfully!")

            # Try forward pass
            output = probe(x)
            print(f"✓ LSTM probe forward pass successful! Output shape: {output.shape}")

        except Exception as e:
            print(f"✗ Error creating/running LSTM probe: {e}")
            print(f"Error type: {type(e)}")
            raise

    def test_layer_investigation(self) -> None:
        """Investigate what layers are actually available in EfficientNet."""
        # Create EfficientNet model
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            efficientnet_variant="b0",
        )

        print("\n=== EfficientNet Layer Investigation ===")

        # Check if the target layer exists
        target_layer = "model.features.8"
        if hasattr(model, "model") and hasattr(model.model, "features"):
            features = model.model.features
            print(f"Features module has {len(features)} submodules")

            if len(features) > 8:
                layer_8 = features[8]
                print(f"Layer 8 type: {type(layer_8)}")
                print(f"Layer 8: {layer_8}")

                if hasattr(layer_8, "out_channels"):
                    print(f"Layer 8 output channels: {layer_8.out_channels}")
                if hasattr(layer_8, "kernel_size"):
                    print(f"Layer 8 kernel size: {layer_8.kernel_size}")
            else:
                print(
                    f"Target layer '{target_layer}' does not exist. Features has "
                    f"{len(features)} submodules"
                )

                # Show what layers are available
                print("Available feature layers:")
                for i, layer in enumerate(features):
                    if hasattr(layer, "out_channels"):
                        print(
                            f"  {i}: {type(layer).__name__} with "
                            f"{layer.out_channels} output channels"
                        )
        else:
            print("Model structure is different than expected")
            print(f"Model has attributes: {dir(model)}")
            if hasattr(model, "model"):
                print(f"Model.model has attributes: {dir(model.model)}")

        print("\n=== Audio Processor Investigation ===")
        if hasattr(model, "audio_processor"):
            print(f"Model has audio_processor: {model.audio_processor}")
            if model.audio_processor is not None:
                print(f"Audio processor attributes: {dir(model.audio_processor)}")
                if hasattr(model.audio_processor, "target_length_seconds"):
                    print(
                        f"target_length_seconds: "
                        f"{model.audio_processor.target_length_seconds}"
                    )
                if hasattr(model.audio_processor, "sample_rate"):
                    print(f"sample_rate: {model.audio_processor.sample_rate}")
                if hasattr(model.audio_processor, "sr"):
                    print(f"sr: {model.audio_processor.sr}")
        else:
            print("Model has no audio_processor attribute")

        print("\n=== Audio Config Investigation ===")
        if hasattr(model, "audio_config"):
            print(f"Model has audio_config: {model.audio_config}")
            if model.audio_config is not None:
                print(f"Audio config attributes: {dir(model.audio_config)}")
                if hasattr(model.audio_config, "sample_rate"):
                    print(f"sample_rate: {model.audio_config.sample_rate}")
        else:
            print("Model has no audio_config attribute")

    def test_lstm_probe_target_layers_all(self) -> None:
        """Test LSTM probe with target_layers='all' to reproduce the original error."""
        # Create EfficientNet model
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            efficientnet_variant="b0",
        )

        # Create dummy audio input
        x = torch.randn(1, 16000)

        # Test with target_layers='all' (this was the failing case)
        try:
            print("\n=== Testing LSTM probe with target_layers='all' ===")

            # First, discover and get available layers
            model._discover_linear_layers()
            available_layers = model._get_all_linear_layers()
            print(f"Available layers: {available_layers}")
            print(f"Number of layers: {len(available_layers)}")

            # Create LSTM probe with target_layers='all'
            probe = LSTMProbe(
                base_model=model,
                layers=available_layers,  # Use all available layers
                num_classes=50,  # Same as in the error case
                device="cpu",
                feature_mode=False,
                aggregation="none",  # This was the failing case
                lstm_hidden_size=256,
                num_layers=2,
                bidirectional=False,  # Same as in the error case
                max_sequence_length=1000,
                use_positional_encoding=False,
                target_length=16000,
            )
            print("✓ LSTM probe created successfully with target_layers='all'!")

            # Try forward pass
            output = probe(x)
            print(f"✓ LSTM probe forward pass successful! Output shape: {output.shape}")
            print("Expected output shape: [1, 50] (batch_size=1, num_classes=50)")

            if output.shape == (1, 50):
                print("✓ Output shape matches expected shape!")
            else:
                print(f"✗ Output shape mismatch: expected (1, 50), got {output.shape}")

        except Exception as e:
            print(f"✗ Error with target_layers='all': {e}")
            print(f"Error type: {type(e)}")
            raise


if __name__ == "__main__":
    # Run tests manually for debugging
    test_instance = TestLSTMProbeEmbeddings()

    print("=== Testing LSTM Probe Embedding Extraction ===\n")

    print("1. Testing EfficientNet embedding extraction...")
    test_instance.test_efficientnet_embedding_extraction_debug()

    print("\n2. Testing LSTM probe creation...")
    test_instance.test_lstm_probe_creation_with_embeddings()

    print("\n3. Investigating EfficientNet layers...")
    test_instance.test_layer_investigation()

    print("\n4. Testing LSTM probe with target_layers='all'...")
    test_instance.test_lstm_probe_target_layers_all()

    print("\n=== All tests completed ===")
