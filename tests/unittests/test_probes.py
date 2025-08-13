"""Tests for the new probe system."""

import pytest
import torch

from representation_learning.configs import ProbeConfig
from representation_learning.models.probes import get_probe


class TestProbeSystem:
    """Test the new probe system."""

    def test_linear_probe_creation(self) -> None:
        """Test that linear probe can be created."""
        probe_config = ProbeConfig(
            name="test_linear",
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "classifier")

    def test_mlp_probe_creation(self) -> None:
        """Test that MLP probe can be created."""
        probe_config = ProbeConfig(
            name="test_mlp",
            probe_type="mlp",
            aggregation="concat",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.2,
            activation="gelu",
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=1024,  # 2 layers * 512 dim = 1024 when concatenated
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "mlp")

    def test_lstm_probe_creation(self) -> None:
        """Test that LSTM probe can be created."""
        probe_config = ProbeConfig(
            name="test_lstm",
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=2,
            bidirectional=True,
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "lstm")

    def test_attention_probe_creation(self) -> None:
        """Test that attention probe can be created."""
        probe_config = ProbeConfig(
            name="test_attention",
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=2,
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "attention_layers")

    def test_transformer_probe_creation(self) -> None:
        """Test that transformer probe can be created."""
        probe_config = ProbeConfig(
            name="test_transformer",
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=8,
            attention_dim=512,
            num_layers=3,
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "transformer")

    def test_linear_probe_forward(self) -> None:
        """Test that linear probe forward pass works."""
        probe_config = ProbeConfig(
            name="test_linear",
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test forward pass
        x = torch.randn(4, 512)  # batch_size=4, embedding_dim=512
        output = probe(x)

        assert output.shape == (4, 10)  # batch_size=4, num_classes=10
        assert not torch.isnan(output).any()

    def test_mlp_probe_forward(self) -> None:
        """Test that MLP probe forward pass works."""
        probe_config = ProbeConfig(
            name="test_mlp",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.1,
            activation="relu",
        )

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test forward pass
        x = torch.randn(4, 512)  # batch_size=4, embedding_dim=512
        output = probe(x)

        assert output.shape == (4, 10)  # batch_size=4, num_classes=10
        assert not torch.isnan(output).any()

    def test_invalid_probe_type(self) -> None:
        """Test that invalid probe type raises error."""
        # Create a probe config with a valid type first, then modify it to test
        # get_probe
        probe_config = ProbeConfig(
            name="test_invalid",
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        # Modify the probe_type to test the get_probe function
        probe_config.probe_type = "invalid_type"

        with pytest.raises(
            NotImplementedError, match="Probe type 'invalid_type' is not implemented"
        ):
            get_probe(
                probe_config=probe_config,
                base_model=None,
                num_classes=10,
                device="cpu",
                feature_mode=True,
                input_dim=512,
            )

    def test_missing_mlp_params(self) -> None:
        """Test that missing MLP parameters raises error."""
        # Create a probe config with hidden_dims first, then remove it to test
        # get_probe
        probe_config = ProbeConfig(
            name="test_mlp_missing",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],  # Add it first
        )

        # Remove hidden_dims to test the get_probe function
        probe_config.hidden_dims = None

        with pytest.raises(
            ValueError, match="MLP probe requires hidden_dims to be specified"
        ):
            get_probe(
                probe_config=probe_config,
                base_model=None,
                num_classes=10,
                device="cpu",
                feature_mode=True,
                input_dim=512,
            )


if __name__ == "__main__":
    pytest.main([__file__])
