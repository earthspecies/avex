"""Test attention probe with realistic dimensions to catch real-world issues."""

import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel
from representation_learning.models.probes.attention_probe import AttentionProbe


class TestAttentionProbeRealisticDimensions:
    """Test attention probe with realistic dimensions to catch real-world issues."""

    def test_attention_probe_realistic_dimensions(self) -> None:
        """Test with realistic batch sizes and sequence lengths."""
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

        # Create attention probe
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
            use_positional_encoding=True,
            target_length=16000,
        )

        # Test with realistic batch size and sequence length
        batch_size = 4  # Smaller than 32 but still realistic
        seq_length = 16000  # Real audio length

        x = torch.randn(batch_size, seq_length)

        # Test forward pass
        output = attention_probe(x)
        print(
            f"✓ Attention probe forward pass successful! Output shape: {output.shape}"
        )
        assert output.shape == (batch_size, 10)

        # Test with padding mask that has different dimensions
        # This simulates the real error case
        padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

        try:
            output_with_mask = attention_probe(x, padding_mask=padding_mask)
            print(
                f"✓ Attention probe with realistic padding mask successful! "
                f"Output shape: {output_with_mask.shape}"
            )
            assert output_with_mask.shape == (batch_size, 10)
        except Exception as e:
            print(f"✗ Attention probe failed with realistic dimensions: {e}")
            raise

        # Verify temporal weights were created with correct dimensions
        assert hasattr(attention_probe, "temporal_weights")
        # The temporal weights should match the embeddings temporal dimension
        # which is different from the input sequence length
        assert attention_probe.temporal_weights.shape[1] > 0

        print("✓ Attention probe works with realistic dimensions!")

    def test_attention_probe_padding_mask_interpolation(self) -> None:
        """Test that padding mask interpolation works correctly."""
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

        # Create attention probe
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

        # Test with padding mask that has different dimensions than embeddings
        batch_size = 2
        input_seq_length = 16000
        x = torch.randn(batch_size, input_seq_length)

        # Create padding mask with input sequence length
        padding_mask = torch.zeros(batch_size, input_seq_length, dtype=torch.bool)

        # This should trigger the padding mask interpolation
        output = attention_probe(x, padding_mask=padding_mask)
        print(f"✓ Padding mask interpolation successful! Output shape: {output.shape}")
        assert output.shape == (batch_size, 10)

        print("✓ Padding mask interpolation works correctly!")


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestAttentionProbeRealisticDimensions()
    test_instance.test_attention_probe_realistic_dimensions()
    test_instance.test_attention_probe_padding_mask_interpolation()
    print("\n🎉 All realistic dimension tests passed!")
