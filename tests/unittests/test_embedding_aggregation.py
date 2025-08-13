"""Tests for the new embedding aggregation functionality."""

import pytest
import torch

from representation_learning.configs import ProbeConfig
from representation_learning.models.probes import get_probe


class TestEmbeddingAggregation:
    """Test the new embedding aggregation functionality."""

    def test_linear_probe_with_different_aggregations(self) -> None:
        """Test that linear probe works with different aggregation methods."""
        # Test mean aggregation
        probe_config_mean = ProbeConfig(
            name="test_mean",
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe_mean = get_probe(
            probe_config=probe_config_mean,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test max aggregation
        probe_config_max = ProbeConfig(
            name="test_max",
            probe_type="linear",
            aggregation="max",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe_max = get_probe(
            probe_config=probe_config_max,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test concat aggregation
        probe_config_concat = ProbeConfig(
            name="test_concat",
            probe_type="linear",
            aggregation="concat",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
        )

        probe_concat = get_probe(
            probe_config=probe_config_concat,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=1024,  # 2 layers * 512 dim
        )

        assert probe_mean is not None
        assert probe_max is not None
        assert probe_concat is not None

        # Test forward pass with mean aggregation
        x = torch.randn(4, 512)
        output_mean = probe_mean(x)
        assert output_mean.shape == (4, 10)

        # Test forward pass with max aggregation
        output_max = probe_max(x)
        assert output_max.shape == (4, 10)

        # Test forward pass with concat aggregation
        x_concat = torch.randn(4, 1024)
        output_concat = probe_concat(x_concat)
        assert output_concat.shape == (4, 10)

    def test_mlp_probe_with_different_aggregations(self) -> None:
        """Test that MLP probe works with different aggregation methods."""
        # Test mean aggregation
        probe_config_mean = ProbeConfig(
            name="test_mlp_mean",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
        )

        probe_mean = get_probe(
            probe_config=probe_config_mean,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test max aggregation
        probe_config_max = ProbeConfig(
            name="test_mlp_max",
            probe_type="mlp",
            aggregation="max",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
        )

        probe_max = get_probe(
            probe_config=probe_config_max,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        assert probe_mean is not None
        assert probe_max is not None

        # Test forward pass
        x = torch.randn(4, 512)
        output_mean = probe_mean(x)
        output_max = probe_max(x)

        assert output_mean.shape == (4, 10)
        assert output_max.shape == (4, 10)

    def test_sequence_probes_with_none_aggregation(self) -> None:
        """Test that sequence probes work with 'none' aggregation."""
        # Test LSTM probe with none aggregation
        probe_config_lstm = ProbeConfig(
            name="test_lstm_none",
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=2,
        )

        probe_lstm = get_probe(
            probe_config=probe_config_lstm,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test attention probe with none aggregation
        probe_config_attention = ProbeConfig(
            name="test_attention_none",
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=2,
        )

        probe_attention = get_probe(
            probe_config=probe_config_attention,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        assert probe_lstm is not None
        assert probe_attention is not None

        # Test forward pass with sequence input
        x_lstm = torch.randn(4, 10, 512)  # batch_size=4, seq_len=10, features=512
        x_attention = torch.randn(4, 10, 256)  # batch_size=4, seq_len=10, features=256

        output_lstm = probe_lstm(x_lstm)
        output_attention = probe_attention(x_attention)

        assert output_lstm.shape == (4, 10)
        assert output_attention.shape == (4, 10)

    def test_invalid_aggregation_method(self) -> None:
        """Test that invalid aggregation method raises error."""
        # Create a valid probe config first, then modify it to test the probe's
        # aggregation logic
        probe_config = ProbeConfig(
            name="test_invalid",
            probe_type="linear",
            aggregation="mean",  # Start with valid aggregation
            input_processing="pooled",
            target_layers=[
                "layer_8",
                "layer_12",
            ],  # Multiple layers to trigger aggregation
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=1024,  # 2 layers * 512 dim
        )

        # Modify the aggregation method to test the probe's validation
        probe.aggregation = "invalid_method"

        # The aggregation error should occur during forward pass
        x = torch.randn(4, 1024)
        with pytest.raises(
            ValueError, match="Unknown aggregation method: invalid_method"
        ):
            probe(x)


if __name__ == "__main__":
    pytest.main([__file__])
