"""Test that the critical attention probe fixes work correctly."""

import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel
from representation_learning.models.probes.attention_probe import AttentionProbe


class TestAttentionProbeCriticalFixes:
    """Test that the critical attention probe fixes work correctly."""

    def test_attention_probe_critical_fixes(self) -> None:
        """Test that all critical fixes work correctly."""
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

        # Create attention probe with all the fixes
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
            use_positional_encoding=True,  # Test positional encoding
            target_length=16000,
        )

        # Create dummy audio input
        x = torch.randn(1, 16000)

        # Test forward pass
        output = attention_probe(x)
        print(
            f"✓ Attention probe forward pass successful! Output shape: {output.shape}"
        )

        # Verify the probe was created successfully
        assert output.shape == (1, 10)

        # Test with padding mask
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)  # 4 time steps, no padding
        output_with_mask = attention_probe(x, padding_mask=padding_mask)
        print(
            f"✓ Attention probe with padding mask successful! "
            f"Output shape: {output_with_mask.shape}"
        )

        # Verify temporal weights were created
        assert hasattr(attention_probe, "temporal_weights")
        assert attention_probe.temporal_weights.shape[1] > 0  # Should have time dim

        # Verify positional encoding was created
        assert attention_probe.pos_encoding is not None
        assert attention_probe.pos_encoding.shape[1] == 1000  # max_sequence_length

        print("✓ All critical attention probe fixes working correctly!")


if __name__ == "__main__":
    # Run test manually
    test_instance = TestAttentionProbeCriticalFixes()
    test_instance.test_attention_probe_critical_fixes()
    print("\n🎉 All critical attention probe fixes passed!")
