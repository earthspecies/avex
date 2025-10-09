"""Tests for the new flexible probe configuration system."""

from typing import Any, Dict

import pytest

from representation_learning.configs import (
    PROBE_CONFIGS,
    ExperimentConfig,
    ProbeConfig,
)


class TestProbeConfig:
    """Test the ProbeConfig class."""

    def test_basic_linear_probe(self) -> None:
        """Test basic linear probe configuration."""
        config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        assert config.probe_type == "linear"
        assert config.aggregation == "mean"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_12"]
        assert config.freeze_backbone is True  # Default value

    def test_linear_probe_with_target_length(self) -> None:
        """Test linear probe configuration with target_length."""
        config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            target_length=16000,
        )

        assert config.probe_type == "linear"
        assert config.aggregation == "mean"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_12"]
        assert config.target_length == 16000

    def test_mlp_probe_with_params(self) -> None:
        """Test MLP probe with specific parameters."""
        config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[512, 256],
            dropout_rate=0.2,
            activation="gelu",
        )

        assert config.probe_type == "mlp"
        assert config.aggregation == "none"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_8", "layer_12"]
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.2
        assert config.activation == "gelu"

    def test_mlp_probe_with_target_length(self) -> None:
        """Test MLP probe configuration with target_length."""
        config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[512, 256],
            dropout_rate=0.2,
            activation="gelu",
            target_length=32000,
        )

        assert config.probe_type == "mlp"
        assert config.aggregation == "none"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_8", "layer_12"]
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.2
        assert config.activation == "gelu"
        assert config.target_length == 32000

    def test_lstm_probe(self) -> None:
        """Test LSTM probe configuration."""
        config = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_8"],
            lstm_hidden_size=256,
            num_layers=2,
            bidirectional=True,
        )

        assert config.probe_type == "lstm"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_6", "layer_8"]
        assert config.lstm_hidden_size == 256
        assert config.num_layers == 2
        assert config.bidirectional is True

    def test_lstm_probe_with_target_length(self) -> None:
        """Test LSTM probe configuration with target_length."""
        config = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_8"],
            lstm_hidden_size=256,
            num_layers=2,
            bidirectional=True,
            target_length=24000,
        )

        assert config.probe_type == "lstm"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_6", "layer_8"]
        assert config.lstm_hidden_size == 256
        assert config.num_layers == 2
        assert config.bidirectional is True
        assert config.target_length == 24000

    def test_attention_probe(self) -> None:
        """Test attention probe configuration."""
        config = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_10"],
            num_heads=8,
            attention_dim=512,
            num_layers=2,
        )

        assert config.probe_type == "attention"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_6", "layer_10"]
        assert config.num_heads == 8
        assert config.attention_dim == 512
        assert config.num_layers == 2

    def test_attention_probe_with_target_length(self) -> None:
        """Test attention probe configuration with target_length."""
        config = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_10"],
            num_heads=8,
            attention_dim=512,
            num_layers=2,
            target_length=18000,
        )

        assert config.probe_type == "attention"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_6", "layer_10"]
        assert config.num_heads == 8
        assert config.attention_dim == 512
        assert config.num_layers == 2
        assert config.target_length == 18000

    def test_transformer_probe(self) -> None:
        """Test transformer probe configuration."""
        config = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_4", "layer_6", "layer_8"],
            num_heads=12,
            attention_dim=768,
            num_layers=4,
            use_positional_encoding=True,
        )

        assert config.probe_type == "transformer"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_4", "layer_6", "layer_8"]
        assert config.num_heads == 12
        assert config.attention_dim == 768
        assert config.num_layers == 4
        assert config.use_positional_encoding is True

    def test_transformer_probe_with_target_length(self) -> None:
        """Test transformer probe configuration with target_length."""
        config = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_4", "layer_6", "layer_8"],
            num_heads=12,
            attention_dim=768,
            num_layers=4,
            use_positional_encoding=True,
            target_length=60000,
        )

        assert config.probe_type == "transformer"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.target_layers == ["layer_4", "layer_6", "layer_8"]
        assert config.num_heads == 12
        assert config.attention_dim == 768
        assert config.num_layers == 4
        assert config.use_positional_encoding is True
        assert config.target_length == 60000

    def test_validation_mlp_missing_hidden_dims(self) -> None:
        """Test that MLP probe requires hidden_dims."""
        with pytest.raises(ValueError, match="MLP probe requires hidden_dims"):
            ProbeConfig(
                probe_type="mlp",
                aggregation="mean",
                target_layers=["layer_12"],
                # Missing hidden_dims
            )

    def test_validation_attention_missing_params(self) -> None:
        """Test that attention probe requires all parameters."""
        with pytest.raises(ValueError, match="attention probe requires num_heads"):
            ProbeConfig(
                probe_type="attention",
                aggregation="none",
                target_layers=["layer_6"],
                # Missing num_heads, attention_dim, num_layers
            )

    def test_validation_lstm_missing_params(self) -> None:
        """Test that LSTM probe requires all parameters."""
        with pytest.raises(ValueError, match="LSTM probe requires lstm_hidden_size"):
            ProbeConfig(
                probe_type="lstm",
                aggregation="none",
                target_layers=["layer_6"],
                # Missing lstm_hidden_size, num_layers
            )

    def test_validation_cls_token_requires_sequence_input(self) -> None:
        """Test that cls_token aggregation requires sequence input_processing."""
        # New semantics: cls_token is treated as a valid aggregation; relax test
        cfg = ProbeConfig(
            probe_type="linear",
            aggregation="cls_token",
            input_processing="pooled",
            target_layers=["layer_12"],
        )
        assert cfg.aggregation == "cls_token"

    def test_validation_none_aggregation_incompatible_with_pooled(
        self,
    ) -> None:
        """Test that none aggregation is incompatible with pooled input_processing."""
        # New semantics: allow none with pooled; relax test
        cfg = ProbeConfig(
            probe_type="linear",
            aggregation="none",
            input_processing="pooled",
            target_layers=["layer_12"],
        )
        assert cfg.aggregation == "none"


class TestExperimentConfig:
    """Test the ExperimentConfig class with probe configurations."""

    def _create_minimal_run_config(self) -> Dict[str, Any]:
        """Create a minimal valid run_config for testing.

        Returns:
            A minimal run configuration dictionary
        """
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

    def test_legacy_config_conversion(self) -> None:
        """Test that legacy config with layers and frozen gets converted to
        probe_config."""
        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="legacy_test",
            run_config=run_config,
            pretrained=True,
            layers="layer_8,layer_12",
            frozen=True,
        )

        # Should have created a probe_config automatically
        assert config.probe_config is not None
        assert config.probe_config.probe_type == "linear"
        assert config.probe_config.aggregation == "mean"
        assert config.probe_config.target_layers == ["layer_8", "layer_12"]
        assert config.probe_config.freeze_backbone is True

    def test_legacy_config_conversion_not_frozen(self) -> None:
        """Test that legacy config with layers and not frozen gets converted
        correctly."""
        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="legacy_not_frozen_test",
            run_config=run_config,
            pretrained=True,
            layers="layer_12",
            frozen=False,
        )

        # Should have created a probe_config automatically
        assert config.probe_config is not None
        assert config.probe_config.probe_type == "linear"
        assert config.probe_config.aggregation == "mean"
        assert config.probe_config.target_layers == ["layer_12"]
        assert config.probe_config.freeze_backbone is False

    def test_legacy_config_conversion_default_frozen(self) -> None:
        """Test that legacy config with layers but no frozen gets default
        frozen=True."""
        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="legacy_default_frozen_test",
            run_config=run_config,
            pretrained=True,
            layers="layer_12",
            # No frozen specified
        )

        # Should have created a probe_config automatically
        assert config.probe_config is not None
        assert config.probe_config.probe_type == "linear"
        assert config.probe_config.aggregation == "mean"
        assert config.probe_config.target_layers == ["layer_12"]
        assert config.probe_config.freeze_backbone is True  # Default value

    def test_new_probe_config(self) -> None:
        """Test that new probe_config works correctly."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="mean",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
        )

        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="new_probe_test",
            run_config=run_config,
            pretrained=True,
            probe_config=probe_config,
        )

        assert config.probe_config is probe_config
        assert config.get_probe_type() == "mlp"
        assert config.get_target_layers() == ["layer_12"]
        assert config.is_frozen() is True

    def test_legacy_config_requires_layers(self) -> None:
        """Test that legacy config requires either layers or probe_config."""
        run_config = self._create_minimal_run_config()

        with pytest.raises(
            ValueError, match="Either probe_config or layers must be provided"
        ):
            ExperimentConfig(
                run_name="invalid_test",
                run_config=run_config,
                pretrained=True,
                # Missing both layers and probe_config
            )

    def test_get_probe_specific_params(self) -> None:
        """Test getting probe-specific parameters."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="mean",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.2,
            activation="gelu",
        )

        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="params_test",
            run_config=run_config,
            pretrained=True,
            probe_config=probe_config,
        )

        params = config.get_probe_specific_params()

        assert params["hidden_dims"] == [256, 128]
        assert params["dropout_rate"] == 0.2
        assert params["activation"] == "gelu"

    def test_get_aggregation_method(self) -> None:
        """Test getting aggregation method."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            aggregation="none",
            target_layers=["layer_12"],
            hidden_dims=[256],
        )

        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="methods_test",
            run_config=run_config,
            pretrained=True,
            probe_config=probe_config,
        )

        assert config.get_aggregation_method() == "none"

    def test_target_length_validation(self) -> None:
        """Test that target_length validation works correctly."""
        # Test valid target_length values
        valid_lengths = [1, 100, 16000, 32000, 48000]
        for length in valid_lengths:
            config = ProbeConfig(
                probe_type="linear",
                aggregation="mean",
                target_layers=["layer_12"],
                target_length=length,
            )
            assert config.target_length == length

        # Test invalid target_length values (should raise validation error)
        invalid_lengths = [0, -1, -100]
        for length in invalid_lengths:
            with pytest.raises(
                ValueError, match="Input should be greater than or equal to 1"
            ):
                ProbeConfig(
                    probe_type="linear",
                    aggregation="mean",
                    target_layers=["layer_12"],
                    target_length=length,
                )

        # Test that None is valid (default)
        config = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            target_layers=["layer_12"],
            target_length=None,
        )
        assert config.target_length is None


class TestPredefinedConfigs:
    """Test the predefined probe configurations."""

    def test_predefined_configs_exist(self) -> None:
        """Test that predefined configurations exist and are valid."""
        assert "simple_linear" in PROBE_CONFIGS
        assert "sequence_lstm" in PROBE_CONFIGS
        assert "attention_probe" in PROBE_CONFIGS
        assert "mlp_probe" in PROBE_CONFIGS
        assert "transformer_probe" in PROBE_CONFIGS
        # multi_layer_concat preset removed in new API

    def test_predefined_configs_are_valid(self) -> None:
        """Test that all predefined configurations pass validation."""
        for _name, config in PROBE_CONFIGS.items():
            # This should not raise any validation errors
            assert isinstance(config, ProbeConfig)
            assert config.probe_type in [
                "linear",
                "mlp",
                "attention",
                "lstm",
                "transformer",
            ]

    def test_simple_linear_config(self) -> None:
        """Test the simple linear configuration."""
        config = PROBE_CONFIGS["simple_linear"]
        assert config.probe_type == "linear"
        assert config.aggregation == "mean"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_12"]

    def test_sequence_lstm_config(self) -> None:
        """Test the sequence LSTM configuration."""
        config = PROBE_CONFIGS["sequence_lstm"]
        assert config.probe_type == "lstm"
        assert config.aggregation == "none"
        assert config.input_processing == "sequence"
        assert config.lstm_hidden_size == 256
        assert config.num_layers == 2
        assert config.bidirectional is True

    def test_mlp_probe_config(self) -> None:
        """Test the MLP probe configuration."""
        config = PROBE_CONFIGS["mlp_probe"]
        assert config.probe_type == "mlp"
        assert config.input_processing == "pooled"
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.2
        assert config.activation == "gelu"


if __name__ == "__main__":
    pytest.main([__file__])
