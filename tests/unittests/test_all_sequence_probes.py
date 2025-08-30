"""Test that all sequence-based probes can be created successfully with EfficientNet."""

import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel
from representation_learning.models.probes.attention_probe import AttentionProbe
from representation_learning.models.probes.lstm_probe import LSTMProbe
from representation_learning.models.probes.transformer_probe import TransformerProbe


class TestAllSequenceProbes:
    """Test that all sequence-based probes can be created and run successfully."""

    def test_all_sequence_probes_creation(self) -> None:
        """Test that all three sequence-based probes can be created successfully."""
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

        # Test LSTM probe
        print("Testing LSTM probe creation...")
        try:
            lstm_probe = LSTMProbe(
                base_model=model,
                layers=["model.features.8"],
                num_classes=10,
                device="cpu",
                feature_mode=False,
                lstm_hidden_size=256,
                num_layers=2,
                bidirectional=True,
                max_sequence_length=1000,
                use_positional_encoding=False,
                target_length=16000,
            )
            print("✓ LSTM probe created successfully!")

            # Test forward pass
            output = lstm_probe(x)
            print(f"✓ LSTM probe forward pass successful! Output shape: {output.shape}")

        except Exception as e:
            print(f"✗ LSTM probe failed: {e}")
            raise

        # Test Attention probe
        print("\nTesting Attention probe creation...")
        try:
            attention_probe = AttentionProbe(
                base_model=model,
                layers=["model.features.8"],
                num_classes=10,
                device="cpu",
                feature_mode=False,
                attention_dim=256,
                num_heads=8,
                num_layers=2,
                dropout_rate=0.1,
                max_sequence_length=1000,
                use_positional_encoding=False,
                target_length=16000,
            )
            print("✓ Attention probe created successfully!")

            # Test forward pass
            output = attention_probe(x)
            print(
                f"✓ Attention probe forward pass successful! "
                f"Output shape: {output.shape}"
            )

        except Exception as e:
            print(f"✗ Attention probe failed: {e}")
            raise

        # Test Transformer probe
        print("\nTesting Transformer probe creation...")
        try:
            transformer_probe = TransformerProbe(
                base_model=model,
                layers=["model.features.8"],
                num_classes=10,
                device="cpu",
                feature_mode=False,
                attention_dim=256,
                num_heads=8,
                num_layers=2,
                dropout_rate=0.1,
                max_sequence_length=1000,
                use_positional_encoding=False,
                target_length=16000,
            )
            print("✓ Transformer probe created successfully!")

            # Test forward pass
            output = transformer_probe(x)
            print(
                f"✓ Transformer probe forward pass successful! "
                f"Output shape: {output.shape}"
            )

        except Exception as e:
            print(f"✗ Transformer probe failed: {e}")
            raise

        print("\n🎉 All sequence-based probes created and tested successfully!")


if __name__ == "__main__":
    # Run test manually
    test_instance = TestAllSequenceProbes()
    test_instance.test_all_sequence_probes_creation()
