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
            name="test_linear",
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        assert config.name == "test_linear"
        assert config.probe_type == "linear"
        assert config.aggregation == "mean"
        assert config.input_processing == "pooled"
        assert config.target_layers == ["layer_12"]
        assert config.freeze_backbone is True
        assert config.learning_rate == 1e-3

    def test_mlp_probe_with_params(self) -> None:
        """Test MLP probe with specific parameters."""
        config = ProbeConfig(
            name="test_mlp",
            probe_type="mlp",
            aggregation="concat",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
            hidden_dims=[512, 256],
            dropout_rate=0.2,
            activation="gelu",
        )

        assert config.probe_type == "mlp"
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.2
        assert config.activation == "gelu"

    def test_lstm_probe(self) -> None:
        """Test LSTM probe configuration."""
        config = ProbeConfig(
            name="test_lstm",
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_8"],
            lstm_hidden_size=256,
            num_layers=2,
            bidirectional=True,
        )

        assert config.probe_type == "lstm"
        assert config.lstm_hidden_size == 256
        assert config.num_layers == 2
        assert config.bidirectional is True

    def test_attention_probe(self) -> None:
        """Test attention probe configuration."""
        config = ProbeConfig(
            name="test_attention",
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_6", "layer_10"],
            num_heads=8,
            attention_dim=512,
            num_layers=2,
        )

        assert config.probe_type == "attention"
        assert config.num_heads == 8
        assert config.attention_dim == 512
        assert config.num_layers == 2

    def test_transformer_probe(self) -> None:
        """Test transformer probe configuration."""
        config = ProbeConfig(
            name="test_transformer",
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
        assert config.num_heads == 12
        assert config.attention_dim == 768
        assert config.num_layers == 4
        assert config.use_positional_encoding is True

    def test_validation_mlp_missing_hidden_dims(self) -> None:
        """Test that MLP probe requires hidden_dims."""
        with pytest.raises(ValueError, match="MLP probe requires hidden_dims"):
            ProbeConfig(
                name="invalid_mlp",
                probe_type="mlp",
                aggregation="mean",
                input_processing="pooled",
                target_layers=["layer_12"],
                # Missing hidden_dims
            )

    def test_validation_attention_missing_params(self) -> None:
        """Test that attention probe requires all parameters."""
        with pytest.raises(ValueError, match="attention probe requires num_heads"):
            ProbeConfig(
                name="invalid_attention",
                probe_type="attention",
                aggregation="none",
                input_processing="sequence",
                target_layers=["layer_6"],
                # Missing num_heads, attention_dim, num_layers
            )

    def test_validation_lstm_missing_params(self) -> None:
        """Test that LSTM probe requires all parameters."""
        with pytest.raises(ValueError, match="LSTM probe requires lstm_hidden_size"):
            ProbeConfig(
                name="invalid_lstm",
                probe_type="lstm",
                aggregation="none",
                input_processing="sequence",
                target_layers=["layer_6"],
                # Missing lstm_hidden_size, num_layers
            )

    def test_validation_cls_token_requires_sequence(self) -> None:
        """Test that cls_token aggregation requires sequence input_processing."""
        with pytest.raises(
            ValueError, match="cls_token aggregation requires sequence input_processing"
        ):
            ProbeConfig(
                name="invalid_cls_token",
                probe_type="linear",
                aggregation="cls_token",
                input_processing="pooled",  # Should be "sequence"
                target_layers=["layer_12"],
            )

    def test_validation_none_aggregation_incompatible_with_pooled(self) -> None:
        """Test that none aggregation is incompatible with pooled input_processing."""
        with pytest.raises(
            ValueError,
            match="none aggregation is incompatible with pooled input_processing",
        ):
            ProbeConfig(
                name="invalid_none_pooled",
                probe_type="linear",
                aggregation="none",
                input_processing="pooled",  # Should not be "pooled"
                target_layers=["layer_12"],
            )

    def test_validation_sequence_probes_require_sequence_input(self) -> None:
        """Test that sequence-based probes require sequence input_processing."""
        with pytest.raises(
            ValueError, match="lstm probe requires sequence or none input_processing"
        ):
            ProbeConfig(
                name="invalid_lstm_input",
                probe_type="lstm",
                aggregation="mean",  # Use mean to avoid aggregation validation error
                input_processing="pooled",  # Should be "sequence" or "none"
                target_layers=["layer_6"],
                lstm_hidden_size=256,
                num_layers=2,
            )


class TestExperimentConfig:
    """Test the ExperimentConfig class with probe configurations."""

    def _create_minimal_run_config(self) -> Dict[str, Any]:
        """Create a minimal valid run_config for testing.

        Returns:
            A minimal run configuration dictionary
        """
        return {
            "model_spec": {"name": "test_model", "pretrained": True, "device": "cuda"},
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

    def test_legacy_config_migration(self) -> None:
        """Test that legacy layers and frozen fields are automatically converted."""
        # Create a minimal mock run config
        mock_run_config = {
            "model_spec": {"name": "test_model"},
            "training_params": {
                "train_epochs": 10,
                "lr": 1e-3,
                "batch_size": 8,
                "optimizer": "adamw",
            },
            "dataset_config": {"train_datasets": [{"dataset_name": "test"}]},
            "output_dir": "./test_output",
            "loss_function": "cross_entropy",
        }

        config = ExperimentConfig(
            run_name="legacy_test",
            run_config=mock_run_config,
            pretrained=True,
            layers="layer_8,layer_12",
            frozen=True,
        )

        # Should have created a probe_config automatically
        assert config.probe_config is not None
        assert config.probe_config.probe_type == "linear"
        assert config.probe_config.aggregation == "mean"
        assert config.probe_config.input_processing == "pooled"
        assert config.probe_config.target_layers == ["layer_8", "layer_12"]
        assert config.probe_config.freeze_backbone is True

    def test_new_probe_config(self) -> None:
        """Test that new probe_config works correctly."""
        probe_config = ProbeConfig(
            name="test_probe",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
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

    def test_get_effective_training_params(self) -> None:
        """Test getting effective training parameters with overrides."""
        from representation_learning.configs import TrainingParams

        global_params = TrainingParams(
            train_epochs=10, lr=1e-3, batch_size=8, optimizer="adamw", weight_decay=0.01
        )

        probe_config = ProbeConfig(
            name="test_probe",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],  # Required for MLP probe
            learning_rate=5e-4,
            batch_size=4,
            train_epochs=15,
        )

        run_config = self._create_minimal_run_config()

        config = ExperimentConfig(
            run_name="training_test",
            run_config=run_config,
            pretrained=True,
            probe_config=probe_config,
        )

        effective_params = config.get_effective_training_params(global_params)

        assert effective_params.lr == 5e-4  # Overridden
        assert effective_params.batch_size == 4  # Overridden
        assert effective_params.train_epochs == 15  # Overridden
        assert effective_params.optimizer == "adamw"  # Not overridden
        assert effective_params.weight_decay == 0.01  # Not overridden

    def test_get_probe_specific_params(self) -> None:
        """Test getting probe-specific parameters."""
        probe_config = ProbeConfig(
            name="test_probe",
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
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

    def test_get_aggregation_and_input_processing(self) -> None:
        """Test getting aggregation and input processing methods."""
        probe_config = ProbeConfig(
            name="test_probe",
            probe_type="mlp",
            aggregation="concat",
            input_processing="pooled",
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

        assert config.get_aggregation_method() == "concat"
        assert config.get_input_processing_method() == "pooled"


class TestPredefinedConfigs:
    """Test the predefined probe configurations."""

    def test_predefined_configs_exist(self) -> None:
        """Test that predefined configurations exist and are valid."""
        assert "simple_linear" in PROBE_CONFIGS
        assert "sequence_lstm" in PROBE_CONFIGS
        assert "attention_probe" in PROBE_CONFIGS
        assert "mlp_probe" in PROBE_CONFIGS
        assert "transformer_probe" in PROBE_CONFIGS
        assert "multi_layer_concat" in PROBE_CONFIGS

    def test_predefined_configs_are_valid(self) -> None:
        """Test that all predefined configurations pass validation."""
        for name, config in PROBE_CONFIGS.items():
            # This should not raise any validation errors
            assert isinstance(config, ProbeConfig)
            assert config.name == name
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
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.2
        assert config.activation == "gelu"


if __name__ == "__main__":
    pytest.main([__file__])
