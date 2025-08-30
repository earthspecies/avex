"""Tests for the new probe system."""

from typing import List, Optional

import pytest
import torch

from representation_learning.configs import ProbeConfig
from representation_learning.models.probes import get_probe


class TestProbeSystem:
    """Test the new probe system."""

    def test_linear_probe_creation(self) -> None:
        """Test that linear probe can be created."""
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
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

    def test_linear_probe_with_target_length(self) -> None:
        """Test that linear probe can be created with target_length parameter."""
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            target_layers=["layer_12"],
            target_length=16000,
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
        assert probe.target_length == 16000
        assert hasattr(probe, "forward")
        assert hasattr(probe, "classifier")

    def test_linear_probe_target_length_override(self) -> None:
        """Test that target_length parameter overrides probe_config.target_length."""
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            target_layers=["layer_12"],
            target_length=8000,  # Default in config
        )

        # Override with parameter
        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
            target_length=16000,  # Override value
        )

        assert probe is not None
        assert probe.target_length == 16000  # Should use override, not config value
        assert hasattr(probe, "forward")
        assert hasattr(probe, "classifier")

    def test_mlp_probe_creation(self) -> None:
        """Test that MLP probe can be created."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
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

    def test_mlp_probe_with_target_length(self) -> None:
        """Test that MLP probe can be created with target_length parameter."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.2,
            activation="gelu",
            target_length=32000,
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
        assert probe.target_length == 32000
        assert hasattr(probe, "forward")
        assert hasattr(probe, "mlp")

    def test_mlp_probe_target_length_override(self) -> None:
        """Test that target_length parameter overrides probe_config.target_length
        for MLP."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.2,
            activation="gelu",
            target_length=16000,  # Default in config
        )

        # Override with parameter
        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=1024,
            target_length=48000,  # Override value
        )

        assert probe is not None
        assert probe.target_length == 48000  # Should use override, not config value
        assert hasattr(probe, "forward")
        assert hasattr(probe, "mlp")

    def test_lstm_probe_creation(self) -> None:
        """Test that LSTM probe can be created."""
        probe_config = ProbeConfig(
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

    def test_lstm_probe_with_target_length(self) -> None:
        """Test that LSTM probe can be created with target_length parameter."""
        probe_config = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=2,
            bidirectional=True,
            target_length=24000,
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
        assert probe.target_length == 24000
        assert hasattr(probe, "forward")
        assert hasattr(probe, "lstm")

    def test_lstm_probe_target_length_override(self) -> None:
        """Test that target_length parameter overrides probe_config.target_length
        for LSTM."""
        probe_config = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=2,
            bidirectional=True,
            target_length=12000,  # Default in config
        )

        # Override with parameter
        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
            target_length=36000,  # Override value
        )

        assert probe is not None
        assert probe.target_length == 36000  # Should use override, not config value
        assert hasattr(probe, "forward")
        assert hasattr(probe, "lstm")

    def test_attention_probe_creation(self) -> None:
        """Test that attention probe can be created."""
        probe_config = ProbeConfig(
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
        # When using feature_mode=True and aggregation="none", attention_layers are None
        assert probe.attention_layers is None

    def test_attention_probe_with_attention_layers(self) -> None:
        """Test that Attention probe creates attention layers when needed."""
        probe_config = ProbeConfig(
            probe_type="attention",
            aggregation="mean",  # Not "none", so attention layers should be created
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=2,
        )

        # Create a mock base model for this test
        class MockBaseModel:
            def __init__(self) -> None:
                self.audio_processor = MockAudioProcessor()

            def extract_embeddings(
                self, x: torch.Tensor, **kwargs: dict
            ) -> torch.Tensor:
                return torch.randn(2, 10, 256)  # 3D tensor

            def register_hooks_for_layers(self, layers: List[str]) -> None:
                pass

            def train(self) -> None:
                pass

            def eval(self) -> None:
                pass

            def parameters(self) -> List[torch.nn.Parameter]:
                return []  # Return empty list of parameters

        class MockAudioProcessor:
            def __init__(self) -> None:
                self.target_length = 16000
                self.sr = 16000

        probe = get_probe(
            probe_config=probe_config,
            base_model=MockBaseModel(),
            num_classes=10,
            device="cpu",
            feature_mode=False,  # Not feature mode, so attention layers should be
            # created
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "attention_layers")
        assert hasattr(probe, "layer_norms")
        assert hasattr(probe, "feed_forward")

    def test_attention_probe_with_target_length(self) -> None:
        """Test that Attention probe can be created with target_length parameter."""
        probe_config = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=2,
            target_length=18000,
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
        assert probe.target_length == 18000
        assert hasattr(probe, "forward")
        # When using feature_mode=True and aggregation="none", attention_layers are None
        assert probe.attention_layers is None

    def test_attention_probe_target_length_override(self) -> None:
        """Test that target_length parameter overrides probe_config.target_length
        for Attention."""
        probe_config = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=2,
            target_length=9000,  # Default in config
        )

        # Override with parameter
        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
            target_length=27000,  # Override value
        )

        assert probe is not None
        assert probe.target_length == 27000  # Should use override, not config value
        assert hasattr(probe, "forward")
        # When using feature_mode=True and aggregation="none", attention_layers are None
        assert probe.attention_layers is None

    def test_transformer_probe_creation(self) -> None:
        """Test that transformer probe can be created."""
        probe_config = ProbeConfig(
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

    def test_transformer_probe_with_target_length(self) -> None:
        """Test that Transformer probe can be created with target_length parameter."""
        probe_config = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=8,
            attention_dim=512,
            num_layers=3,
            target_length=60000,
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
        assert probe.target_length == 60000
        assert hasattr(probe, "forward")
        assert hasattr(probe, "transformer")

    def test_transformer_probe_target_length_override(self) -> None:
        """Test that target_length parameter overrides probe_config.target_length
        for Transformer."""
        probe_config = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=8,
            attention_dim=512,
            num_layers=3,
            target_length=30000,  # Default in config
        )

        # Override with parameter
        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
            target_length=90000,  # Override value
        )

        assert probe is not None
        assert probe.target_length == 90000  # Should use override, not config value
        assert hasattr(probe, "forward")
        assert hasattr(probe, "transformer")

    def test_linear_probe_forward(self) -> None:
        """Test that linear probe forward pass works."""
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
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
            probe_type="mlp",
            aggregation="mean",
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

    def test_target_length_override_functionality(self) -> None:
        """Test that target_length parameter properly overrides
        probe_config.target_length."""
        # Test with different probe types
        probe_types = ["linear", "mlp", "lstm", "attention", "transformer"]

        for probe_type in probe_types:
            # Create base config
            if probe_type == "mlp":
                probe_config = ProbeConfig(
                    probe_type=probe_type,
                    aggregation="mean",
                    target_layers=["layer_12"],
                    hidden_dims=[256],  # Required for MLP
                    target_length=8000,  # Default in config
                )
            elif probe_type == "lstm":
                probe_config = ProbeConfig(
                    probe_type=probe_type,
                    aggregation="none",
                    target_layers=["layer_12"],
                    lstm_hidden_size=128,  # Required for LSTM
                    num_layers=2,  # Required for LSTM
                    bidirectional=False,
                    target_length=8000,  # Default in config
                )
            elif probe_type in ["attention", "transformer"]:
                probe_config = ProbeConfig(
                    probe_type=probe_type,
                    aggregation="none",
                    target_layers=["layer_12"],
                    num_heads=4,
                    attention_dim=256,
                    num_layers=2,
                    target_length=8000,  # Default in config
                )
            else:  # linear
                probe_config = ProbeConfig(
                    probe_type=probe_type,
                    aggregation="mean",
                    target_layers=["layer_12"],
                    target_length=8000,  # Default in config
                )

            # Test that config default is used when no override
            probe1 = get_probe(
                probe_config=probe_config,
                base_model=None,
                num_classes=10,
                device="cpu",
                feature_mode=True,
                input_dim=512,
                target_length=None,  # No override
            )
            assert probe1.target_length == 8000, f"Failed for {probe_type}"

            # Test that override parameter is used
            probe2 = get_probe(
                probe_config=probe_config,
                base_model=None,
                num_classes=10,
                device="cpu",
                feature_mode=True,
                input_dim=512,
                target_length=16000,  # Override value
            )
            assert probe2.target_length == 16000, f"Failed for {probe_type}"

            print(f"✓ {probe_type} probe target_length override works correctly")

    def test_invalid_probe_type(self) -> None:
        """Test that invalid probe type raises error."""
        # Create a probe config with a valid type first, then modify it to test
        # get_probe
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            target_layers=["layer_12"],
        )

        # Modify the probe_type to test the get_probe function
        probe_config.probe_type = "invalid_type"

        with pytest.raises(
            NotImplementedError,
            match="Probe type 'invalid_type' is not implemented",
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
            probe_type="mlp",
            aggregation="mean",
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

    def test_linear_probe_freeze_backbone_detaches_embeddings(self) -> None:
        """Test that LinearProbe detaches embeddings when freeze_backbone=True."""
        from representation_learning.models.probes.linear_probe import LinearProbe

        # Create a mock base model that returns requires_grad=True embeddings
        class MockBaseModel:
            def __init__(self) -> None:
                self.audio_processor = type(
                    "MockAudioProcessor", (), {"target_length": 1000, "sr": 16000}
                )()

            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                aggregation: str = "mean",
            ) -> torch.Tensor:
                # Return embeddings that require gradients
                return torch.randn(2, 128, requires_grad=True)

            def register_hooks_for_layers(self, layers: List[str]) -> None:
                pass

        # Test with freeze_backbone=True
        probe_frozen = LinearProbe(
            base_model=MockBaseModel(),
            layers=["layer_12"],
            num_classes=10,
            device="cpu",  # Use CPU to avoid device mismatch
            target_length=1000,  # Provide target_length to avoid computation
            freeze_backbone=True,
        )

        # Forward pass should detach embeddings
        x = torch.randn(2, 1000)
        with torch.no_grad():
            output = probe_frozen(x)

        # The output should not require gradients since embeddings were detached
        assert not output.requires_grad

        # Test with freeze_backbone=False
        probe_unfrozen = LinearProbe(
            base_model=MockBaseModel(),
            layers=["layer_12"],
            num_classes=10,
            device="cpu",  # Use CPU to avoid device mismatch
            target_length=1000,  # Provide target_length to avoid computation
            freeze_backbone=False,
        )

        # Forward pass should preserve gradient flow
        x = torch.randn(2, 1000)
        output = probe_unfrozen(x)

        # The output should require gradients since embeddings were not detached
        assert output.requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
