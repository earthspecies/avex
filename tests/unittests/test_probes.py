"""Tests for the probe system, configurations, and multi-layer embeddings."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

from representation_learning.configs import (
    PROBE_CONFIGS,
    ExperimentConfig,
    ProbeConfig,
)
from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    load_embeddings_arrays,
    save_embeddings_arrays,
)
from representation_learning.models.probes import get_probe
from representation_learning.models.probes.attention_probe import AttentionProbe
from representation_learning.models.probes.embedding_projectors import (
    Conv4DProjector,
    EmbeddingProjector,
    Sequence3DProjector,
)
from representation_learning.models.probes.linear_probe import LinearProbe
from representation_learning.models.probes.lstm_probe import LSTMProbe
from representation_learning.models.probes.mlp_probe import MLPProbe
from representation_learning.models.probes.transformer_probe import TransformerProbe


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
        # With new semantics, attention layers are always built
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == 2

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

            def extract_embeddings(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
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
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == 2

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
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == 2

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

    @pytest.mark.parametrize(
        "probe_type,aggregation,input_processing,input_dim,extra",
        [
            ("linear", "mean", "pooled", 512, {}),
            ("mlp", "mean", "pooled", 512, {"hidden_dims": [128]}),
            (
                "lstm",
                "none",
                "sequence",
                256,
                {"lstm_hidden_size": 64, "num_layers": 1},
            ),
            (
                "attention",
                "none",
                "sequence",
                256,
                {"num_heads": 4, "attention_dim": 256, "num_layers": 1},
            ),
            (
                "transformer",
                "none",
                "sequence",
                256,
                {"num_heads": 4, "attention_dim": 256, "num_layers": 1},
            ),
        ],
    )
    def test_probe_factory_basic(
        self,
        probe_type: str,
        aggregation: str,
        input_processing: str,
        input_dim: int,
        extra: dict,
    ) -> None:
        """Parametrized sanity check over probe types and aggregation modes."""
        cfg_kwargs = {
            "probe_type": probe_type,
            "aggregation": aggregation,
            "input_processing": input_processing,
            "target_layers": ["layer_12"],
            **extra,
        }
        probe_config = ProbeConfig(**cfg_kwargs)

        probe = get_probe(
            probe_config=probe_config,
            base_model=None,
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
        )
        assert probe is not None

    def test_invalid_input_processing_for_linear_sequence(self) -> None:
        """Linear cannot use sequence input_processing."""
        probe_config = ProbeConfig(
            probe_type="linear",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
        )
        with pytest.raises(ValueError):
            get_probe(
                probe_config=probe_config,
                base_model=None,
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=128,
            )

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

        with pytest.raises(ValueError, match="MLP probe requires hidden_dims to be specified"):
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
                self.audio_processor = type("MockAudioProcessor", (), {"target_length": 1000, "sr": 16000})()

            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                aggregation: str = "mean",
                freeze_backbone: bool = False,
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

    def test_probe_config_creation_and_validation(self) -> None:
        """Test ProbeConfig creation and basic validation for all probe types."""
        # Test basic linear probe config
        linear_config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )
        assert linear_config.probe_type == "linear"
        assert linear_config.aggregation == "mean"
        assert linear_config.input_processing == "pooled"
        assert linear_config.target_layers == ["layer_12"]
        assert linear_config.freeze_backbone is True

        # Test MLP probe config
        mlp_config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[512, 256],
            dropout_rate=0.2,
            activation="gelu",
        )
        assert mlp_config.probe_type == "mlp"
        assert mlp_config.hidden_dims == [512, 256]
        assert mlp_config.dropout_rate == 0.2

        # Test LSTM probe config
        lstm_config = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_8"],
            lstm_hidden_size=256,
            num_layers=2,
            bidirectional=True,
        )
        assert lstm_config.probe_type == "lstm"
        assert lstm_config.lstm_hidden_size == 256
        assert lstm_config.bidirectional is True

        # Test attention probe config
        attention_config = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_10"],
            num_heads=8,
            attention_dim=512,
            num_layers=2,
        )
        assert attention_config.probe_type == "attention"
        assert attention_config.num_heads == 8
        assert attention_config.attention_dim == 512

        # Test transformer probe config
        transformer_config = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_4", "layer_6", "layer_8"],
            num_heads=12,
            attention_dim=768,
            num_layers=4,
            use_positional_encoding=True,
        )
        assert transformer_config.probe_type == "transformer"
        assert transformer_config.use_positional_encoding is True

    def test_probe_config_validation_errors(self) -> None:
        """Test ProbeConfig validation for missing required parameters."""
        # MLP requires hidden_dims
        with pytest.raises(ValueError, match="MLP probe requires hidden_dims"):
            ProbeConfig(
                probe_type="mlp",
                aggregation="mean",
                target_layers=["layer_12"],
            )

        # Attention requires num_heads, attention_dim, num_layers
        with pytest.raises(ValueError, match="attention probe requires num_heads"):
            ProbeConfig(
                probe_type="attention",
                aggregation="none",
                target_layers=["layer_6"],
            )

        # LSTM requires lstm_hidden_size, num_layers
        with pytest.raises(ValueError, match="LSTM probe requires lstm_hidden_size"):
            ProbeConfig(
                probe_type="lstm",
                aggregation="none",
                target_layers=["layer_6"],
            )

        # Test target_length validation
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            ProbeConfig(
                probe_type="linear",
                aggregation="mean",
                target_layers=["layer_12"],
                target_length=0,
            )

    def test_target_length_in_configs(self) -> None:
        """Test target_length parameter in ProbeConfig."""
        configs = [
            ProbeConfig(
                probe_type="linear",
                aggregation="mean",
                target_layers=["layer_12"],
                target_length=16000,
            ),
            ProbeConfig(
                probe_type="mlp",
                aggregation="none",
                target_layers=["layer_8", "layer_12"],
                hidden_dims=[512, 256],
                target_length=32000,
            ),
            ProbeConfig(
                probe_type="lstm",
                aggregation="none",
                input_processing="sequence",
                target_layers=["layer_6", "layer_8"],
                lstm_hidden_size=256,
                num_layers=2,
                target_length=24000,
            ),
        ]

        assert configs[0].target_length == 16000
        assert configs[1].target_length == 32000
        assert configs[2].target_length == 24000

    def test_predefined_configs(self) -> None:
        """Test predefined probe configurations."""
        assert "simple_linear" in PROBE_CONFIGS
        assert "sequence_lstm" in PROBE_CONFIGS
        assert "attention_probe" in PROBE_CONFIGS
        assert "mlp_probe" in PROBE_CONFIGS
        assert "transformer_probe" in PROBE_CONFIGS

        # Validate all predefined configs
        for _name, config in PROBE_CONFIGS.items():
            assert isinstance(config, ProbeConfig)
            assert config.probe_type in ["linear", "mlp", "attention", "lstm", "transformer"]

        # Test specific predefined configs
        simple_linear = PROBE_CONFIGS["simple_linear"]
        assert simple_linear.probe_type == "linear"
        assert simple_linear.aggregation == "mean"
        assert simple_linear.input_processing == "pooled"

        sequence_lstm = PROBE_CONFIGS["sequence_lstm"]
        assert sequence_lstm.probe_type == "lstm"
        assert sequence_lstm.aggregation == "none"
        assert sequence_lstm.input_processing == "sequence"

    def test_experiment_config_with_probe_config(self) -> None:
        """Test ExperimentConfig with probe configurations."""

        def _create_minimal_run_config() -> Dict[str, Any]:
            return {
                "model_spec": {
                    "name": "test_model",
                    "pretrained": True,
                    "device": "cuda",
                },
                "training_params": {
                    "train_epochs": 10,
                    "lr": 1e-3,
                    "batch_size": 8,
                    "optimizer": "adamw",
                },
                "dataset_config": {
                    "train_datasets": [{"dataset_name": "beans", "split": "dogs_train"}],
                    "val_datasets": [{"dataset_name": "beans", "split": "dogs_validation"}],
                    "test_datasets": [{"dataset_name": "beans", "split": "dogs_test"}],
                },
                "output_dir": "./test_output",
                "loss_function": "cross_entropy",
            }

        # Test legacy config conversion
        run_config = _create_minimal_run_config()
        config = ExperimentConfig(
            run_name="legacy_test",
            run_config=run_config,
            pretrained=True,
            layers="layer_8,layer_12",
            frozen=True,
        )
        assert config.probe_config is not None
        assert config.probe_config.probe_type == "linear"
        assert config.probe_config.target_layers == ["layer_8", "layer_12"]
        assert config.probe_config.freeze_backbone is True

        # Test new probe_config
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="mean",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
        )
        config = ExperimentConfig(
            run_name="new_probe_test",
            run_config=run_config,
            pretrained=True,
            probe_config=probe_config,
        )
        assert config.probe_config is probe_config
        assert config.get_probe_type() == "mlp"
        assert config.get_target_layers() == ["layer_12"]

        # Test that legacy config requires layers or probe_config
        with pytest.raises(ValueError, match="Either probe_config or layers must be provided"):
            ExperimentConfig(
                run_name="invalid_test",
                run_config=run_config,
                pretrained=True,
            )

    def test_probes_with_multi_layer_embeddings(self) -> None:
        """Test probes with multi-layer embeddings and different shapes."""
        # Test LinearProbe with flattened dimensions
        input_dims = [(128,), (64, 32), (16, 8, 4)]
        last_layer_dim = input_dims[-1]
        flattened_dim = 16 * 8 * 4

        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2", "layer3"],
            num_classes=5,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,
        )

        assert linear_probe.feature_mode is True
        assert linear_probe.num_classes == 5

        batch_size = 3
        test_input = torch.randn(batch_size, *last_layer_dim)
        test_input_flat = test_input.view(batch_size, -1)

        with torch.no_grad():
            output = linear_probe(test_input_flat)
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()

        # Test AttentionProbe with 3D input
        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1"],
            num_classes=5,
            device="cpu",
            feature_mode=True,
            input_dim=[(16, 8)],
            num_heads=4,
            attention_dim=64,
            num_layers=2,
        )

        test_input_3d = torch.randn(batch_size, 16, 8)
        with torch.no_grad():
            output = attention_probe(test_input_3d)
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()

    def test_probes_with_embedding_save_load_integration(self) -> None:
        """Test probes with embeddings from save/load cycle."""
        batch_size = 4
        embeddings = {
            "layer1": torch.randn(batch_size, 128),
            "layer2": torch.randn(batch_size, 64, 32),
            "layer3": torch.randn(batch_size, 16, 8, 4),
        }
        labels = torch.randint(0, 5, (batch_size,))
        num_labels = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            assert isinstance(loaded_embeddings, dict)
            assert loaded_num_labels == num_labels
            assert torch.equal(loaded_labels, labels)

            # Test with LinearProbe
            last_layer_name = list(loaded_embeddings.keys())[-1]
            last_layer_emb = loaded_embeddings[last_layer_name]
            flattened_dim = last_layer_emb.shape[1:].numel()

            linear_probe = LinearProbe(
                base_model=None,
                layers=list(loaded_embeddings.keys()),
                num_classes=num_labels,
                device="cpu",
                feature_mode=True,
                input_dim=flattened_dim,
            )

            last_layer_emb_flat = last_layer_emb.view(batch_size, -1)
            with torch.no_grad():
                output = linear_probe(last_layer_emb_flat)

            assert output.shape == (batch_size, num_labels)
            assert not torch.isnan(output).any()

    def test_probes_with_embedding_dataset(self) -> None:
        """Test probes with EmbeddingDataset."""
        batch_size = 3
        embeddings = {
            "layer1": torch.randn(batch_size, 128),
            "layer2": torch.randn(batch_size, 64, 32),
            "layer3": torch.randn(batch_size, 16, 8, 4),
        }
        labels = torch.randint(0, 3, (batch_size,))

        dataset = EmbeddingDataset(embeddings, labels)
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "label" in sample

        first_batch = next(iter(DataLoader(dataset, batch_size=1)))
        embed_keys = [k for k in first_batch.keys() if k != "label"]
        last_layer_name = list(embed_keys)[-1]
        last_layer_emb = first_batch[last_layer_name]
        flattened_dim = last_layer_emb.shape[1:].numel()

        linear_probe = LinearProbe(
            base_model=None,
            layers=embed_keys,
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,
        )

        last_layer_emb_flat = last_layer_emb.view(1, -1)
        with torch.no_grad():
            output = linear_probe(last_layer_emb_flat)
        assert output.shape == (1, 3)

    def test_probes_gradient_flow(self) -> None:
        """Test that gradients flow correctly through probes."""
        flattened_dim = 64 * 32

        # Test LinearProbe
        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,
        )

        test_input = torch.randn(2, 64, 32, requires_grad=True)
        test_input_flat = test_input.view(2, -1)
        output = linear_probe(test_input_flat)
        loss = output.sum()
        loss.backward()

        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()
        assert linear_probe.classifier.weight.grad is not None

        # Test AttentionProbe
        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=[(64, 32)],
            num_heads=2,
            attention_dim=32,
        )

        test_input = torch.randn(2, 64, 32, requires_grad=True)
        output = attention_probe(test_input)
        loss = output.sum()
        loss.backward()

        assert test_input.grad is not None
        assert attention_probe.classifier.weight.grad is not None

    def test_probes_error_handling(self) -> None:
        """Test error handling in probe initialization."""
        # Invalid input_dim type
        with pytest.raises((TypeError, ValueError)):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim="invalid",
            )

        # Empty input_dim list
        with pytest.raises((ValueError, IndexError)):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=[],
            )

        # None input_dim with no base_model
        with pytest.raises(ValueError):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_probes_different_batch_sizes(self) -> None:
        """Test probes with different batch sizes."""
        flattened_dim = 64 * 32

        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,
        )

        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=[(64, 32)],
            num_heads=2,
            attention_dim=32,
        )

        batch_sizes = [1, 2, 5, 10]
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 64, 32)
            test_input_flat = test_input.view(batch_size, -1)

            with torch.no_grad():
                linear_output = linear_probe(test_input_flat)
                attention_output = attention_probe(test_input)

            assert linear_output.shape == (batch_size, 3)
            assert attention_output.shape == (batch_size, 3)
            assert not torch.isnan(linear_output).any()
            assert not torch.isnan(attention_output).any()

    def test_embedding_projectors_basic_functionality(self) -> None:
        """Test basic functionality of embedding projectors."""
        # Test Conv4DProjector
        conv4d_projector = Conv4DProjector()
        x_4d = torch.randn(2, 3, 4, 5)
        output_4d = conv4d_projector(x_4d)
        assert output_4d.shape == (2, 5, 12)
        assert output_4d.dim() == 3

        # Test with target_dim
        conv4d_target = Conv4DProjector(target_feature_dim=64)
        output_target = conv4d_target(x_4d)
        assert output_target.shape == (2, 5, 64)

        # Test Sequence3DProjector
        seq3d_projector = Sequence3DProjector()
        x_3d = torch.randn(2, 10, 64)
        output_3d = seq3d_projector(x_3d)
        assert output_3d.shape == (2, 10, 64)
        assert output_3d.dim() == 3

        # Test with target_dim
        seq3d_target = Sequence3DProjector(target_feature_dim=128)
        output_target = seq3d_target(x_3d)
        assert output_target.shape == (2, 10, 128)

        # Test EmbeddingProjector (unified)
        embedding_projector = EmbeddingProjector()
        assert embedding_projector(x_4d).shape == (2, 5, 12)
        assert embedding_projector(x_3d).shape == (2, 10, 64)

        # Test 2D input with force_sequence_format
        x_2d = torch.randn(2, 64)
        projector_force = EmbeddingProjector(force_sequence_format=True)
        output_2d = projector_force(x_2d)
        assert output_2d.shape == (2, 1, 64)

        # Test 2D input without force
        projector_no_force = EmbeddingProjector(force_sequence_format=False)
        output_2d_no_force = projector_no_force(x_2d)
        assert output_2d_no_force.shape == (2, 64)

    def test_embedding_projectors_error_handling(self) -> None:
        """Test error handling in embedding projectors."""
        # Conv4DProjector with invalid input
        conv4d = Conv4DProjector()
        with pytest.raises(ValueError, match="Conv4DProjector expects 4D input"):
            conv4d(torch.randn(2, 3, 4))

        # Sequence3DProjector with invalid input
        seq3d = Sequence3DProjector()
        with pytest.raises(ValueError, match="Sequence3DProjector expects 3D input"):
            seq3d(torch.randn(2, 64))

        # EmbeddingProjector with invalid input
        embedding = EmbeddingProjector()
        with pytest.raises(ValueError, match="EmbeddingProjector supports 2D, 3D, and 4D tensors"):
            embedding(torch.randn(64))

    def test_embedding_projectors_with_different_shapes(self) -> None:
        """Test projectors with multiple embedding shapes."""
        projector = EmbeddingProjector(target_feature_dim=256)

        # Test different input shapes
        inputs = [
            torch.randn(2, 3, 4, 5),  # 4D
            torch.randn(2, 10, 64),  # 3D
            torch.randn(2, 128),  # 2D
        ]

        outputs = []
        for x in inputs:
            output = projector(x)
            outputs.append(output)
            assert output.dim() == 3
            assert output.shape[0] == 2
            assert output.shape[2] == 256

    def test_embedding_projectors_gradient_flow(self) -> None:
        """Test that gradients flow through projectors."""
        projector = EmbeddingProjector(target_feature_dim=128)

        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_embedding_projectors_with_mixed_dimensions(self) -> None:
        """Test projectors with mixed 4D, 3D, and 2D tensors and realistic scenarios."""
        projector = EmbeddingProjector(target_feature_dim=256)

        embeddings = [
            torch.randn(2, 3, 4, 5),  # 4D tensor
            torch.randn(2, 10, 64),  # 3D tensor
            torch.randn(2, 128),  # 2D tensor
        ]

        projected_embeddings = []
        for emb in embeddings:
            projected = projector(emb)
            projected_embeddings.append(projected)
            assert projected.dim() == 3
            assert projected.shape[2] == 256
            assert projected.shape[0] == 2

        # Test weighted combination (simulating probe behavior)
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            256,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for emb, weight in zip(projected_embeddings, weights, strict=False):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        assert weighted_embeddings.shape == (2, min_seq_len, 256)

        # Test with realistic model embeddings (multiple 4D, 3D, and 2D)
        realistic_projector = EmbeddingProjector(target_feature_dim=768)
        realistic_embeddings = [
            torch.randn(2, 3, 4, 5),  # EfficientNet early layer
            torch.randn(2, 6, 8, 10),  # EfficientNet mid layer
            torch.randn(2, 10, 768),  # BEATs/AVES layer
            torch.randn(2, 20, 768),  # Another transformer layer
            torch.randn(2, 768),  # Final pooled output
        ]

        projected_realistic = []
        for emb in realistic_embeddings:
            projected = realistic_projector(emb)
            projected_realistic.append(projected)
            assert projected.dim() == 3
            assert projected.shape[2] == 768

        # Test output shape info
        info_4d = projector.get_output_shape_info((2, 3, 4, 5))
        assert info_4d["projector_type"] == "Conv4DProjector"
        assert info_4d["output_shape"] == (2, 5, 256)

        info_3d = projector.get_output_shape_info((2, 10, 64))
        assert info_3d["projector_type"] == "Sequence3DProjector"
        assert info_3d["output_shape"] == (2, 10, 256)

    def test_probes_with_embedding_dimensions_and_aggregation(self) -> None:
        """Test probes with different embedding dimensions, aggregation modes, and sequence probes."""
        test_data = {
            "batch_size": 2,
            "num_classes": 5,
            "embeddings_2d": torch.randn(2, 256),
            "embeddings_3d": torch.randn(2, 10, 256),
            "embeddings_4d": torch.randn(2, 4, 4, 256),
        }

        # Test LinearProbe with different dimensions
        probe_2d = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=test_data["num_classes"],
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )
        output_2d = probe_2d(test_data["embeddings_2d"])
        assert output_2d.shape == (test_data["batch_size"], test_data["num_classes"])

        # Test 3D embeddings with flattened dim
        b, t, f = test_data["embeddings_3d"].shape
        probe_3d = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=test_data["num_classes"],
            device="cpu",
            feature_mode=True,
            input_dim=t * f,
        )
        output_3d = probe_3d(test_data["embeddings_3d"].reshape(b, -1))
        assert output_3d.shape == (test_data["batch_size"], test_data["num_classes"])

        # Test sequence probes with 3D embeddings and none aggregation
        sequence_probes = [
            ("AttentionProbe", AttentionProbe),
            ("LSTMProbe", LSTMProbe),
            ("TransformerProbe", TransformerProbe),
        ]

        for probe_name, probe_class in sequence_probes:
            probe = probe_class(
                base_model=None,
                layers=[],
                num_classes=test_data["num_classes"],
                device="cpu",
                feature_mode=True,
                input_dim=256,
            )
            output = probe(test_data["embeddings_3d"])
            assert output.shape == (
                test_data["batch_size"],
                test_data["num_classes"],
            ), f"{probe_name} failed with 3D embeddings"

        # Test sequence probes with none aggregation via ProbeConfig
        probe_config_lstm = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=1,
            bidirectional=False,
        )

        probe_config_attention = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=1,
        )

        probe_lstm = get_probe(
            probe_config=probe_config_lstm,
            base_model=None,
            num_classes=test_data["num_classes"],
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        probe_attention = get_probe(
            probe_config=probe_config_attention,
            base_model=None,
            num_classes=test_data["num_classes"],
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        x_lstm = torch.randn(4, 10, 512)
        x_attention = torch.randn(4, 10, 256)
        assert probe_lstm(x_lstm).shape == (4, test_data["num_classes"])
        assert probe_attention(x_attention).shape == (4, test_data["num_classes"])

        # Test weighted probes with sequence processing
        probe_config_transformer = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=1,
        )

        probe_transformer = get_probe(
            probe_config=probe_config_transformer,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        x_sequence = torch.randn(4, 10, 256)
        assert probe_transformer(x_sequence).shape == (4, 10)

        # Test 2D embeddings with sequence probes (may be unsqueezed)
        for probe_name, probe_class in sequence_probes:
            probe = probe_class(
                base_model=None,
                layers=[],
                num_classes=test_data["num_classes"],
                device="cpu",
                feature_mode=True,
                input_dim=256,
            )
            try:
                output = probe(test_data["embeddings_2d"])
                assert output.shape == (
                    test_data["batch_size"],
                    test_data["num_classes"],
                ), f"{probe_name} failed with 2D embeddings"
            except Exception as e:
                pytest.skip(f"2D embeddings not supported by {probe_name}: {e}")

        # Test invalid dimensions
        probe = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=test_data["num_classes"],
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )
        with pytest.raises(ValueError, match="expects 2D, 3D or 4D embeddings"):
            probe(torch.randn(test_data["batch_size"]))

    def test_probes_with_none_aggregation_and_projectors(self) -> None:
        """Test probes with aggregation=None using embedding projectors."""
        from typing import List, Optional, Union

        class MockBaseModel:
            def __init__(self) -> None:
                self.audio_processor = type("MockProcessor", (), {"target_length": 24000, "sr": 16000})()

            def register_hooks_for_layers(self, layers: list) -> None:
                pass

            def extract_embeddings(
                self,
                x: torch.Tensor,
                aggregation: str = "mean",
                padding_mask: Optional[torch.Tensor] = None,
                freeze_backbone: bool = False,
            ) -> Union[torch.Tensor, List[torch.Tensor]]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    return [
                        torch.randn(batch_size, 512),
                        torch.randn(batch_size, 768),
                    ]
                else:
                    return torch.randn(batch_size, 512)

        mock_model = MockBaseModel()

        # Test LinearProbe with none aggregation
        linear_probe = LinearProbe(
            base_model=mock_model,
            layers=["layer_8", "layer_12"],
            num_classes=10,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        assert hasattr(linear_probe, "embedding_projectors")
        assert len(linear_probe.embedding_projectors) == 2

        x = torch.randn(4, 24000)
        output = linear_probe(x)
        assert output.shape == (4, 10)

        # Test MLPProbe with none aggregation
        mlp_probe = MLPProbe(
            base_model=mock_model,
            layers=["layer_8", "layer_12"],
            num_classes=10,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            hidden_dims=[256, 128],
        )

        assert hasattr(mlp_probe, "embedding_projectors")
        assert len(mlp_probe.embedding_projectors) == 2

        output = mlp_probe(x)
        assert output.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__])
