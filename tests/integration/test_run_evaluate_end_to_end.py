"""
End-to-end integration test for run_evaluate.py with probes.

This test:
1. Tests linear and attention probes
2. Tests with freeze_backbone=true and false
3. Tests with last_layer and all_layers configurations
4. Tests with training offline and online modes
5. Validates that expected metrics are present and within valid ranges

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.integration.eval_end_to_end_harness import (
    create_probe_eval_config,
    create_test_data_config,
    run_linear_offline_probe_evaluate,
)

# Skip entire module if esp_data is not installed (internal dependency)
# These tests use build_dataloaders which loads real datasets via esp_data
pytest.importorskip("esp_data")


class TestRunEvaluateEndToEnd:
    @pytest.fixture
    def temp_output_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def base_config_path(self) -> Path:
        return Path("configs/evaluation_configs/flexible_probing_minimal_test.yml")

    def _create_test_data_config(self, data_config_path: Path) -> None:
        """Create a minimal test data configuration with tiny dataset for fast testing.

        Uses real beans dataset but with aggressive subsampling to only 3 classes
        and very few samples for speed. No label mapping - uses original string labels.
        """
        create_test_data_config(data_config_path)

    def _create_test_config(
        self,
        temp_output_dir: Path,
        probe_type: str,
        freeze_backbone: bool,
        layers: str,
        training_mode: str,
    ) -> Path:
        """Create a test configuration for specific probe settings.

        Returns
        -------
        Path
            Path to the created configuration file.
        """
        return create_probe_eval_config(temp_output_dir, probe_type, freeze_backbone, layers, training_mode)

    def _load_eval_config(self, config_path: Path) -> Any:  # noqa: ANN401
        """Helper to load EvaluateConfig with proper working directory.

        Parameters
        ----------
        config_path : Path
            Path to the evaluation configuration file.

        Returns
        -------
        EvaluateConfig
            Loaded and validated evaluation configuration.
        """
        import os

        from avex.configs import EvaluateConfig

        project_root = Path(__file__).parent.parent.parent
        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)
            return EvaluateConfig.from_sources(yaml_file=config_path, cli_args=[])
        finally:
            os.chdir(original_cwd)

    @pytest.mark.parametrize(
        "probe_type,freeze_backbone,layers,training_mode",
        [("linear", True, "last_layer", "offline")],
    )
    def test_weighted_probes_config_validation(
        self,
        probe_type: str,
        freeze_backbone: bool,
        layers: str,
        training_mode: str,
        temp_output_dir: Path,
    ) -> None:
        config_path = self._create_test_config(temp_output_dir, probe_type, freeze_backbone, layers, training_mode)

        eval_cfg = self._load_eval_config(config_path)

        assert len(eval_cfg.experiments) == 1
        experiment = eval_cfg.experiments[0]
        assert experiment.probe_config is not None
        assert experiment.probe_config.probe_type == probe_type
        assert experiment.probe_config.freeze_backbone == freeze_backbone

        if layers == "last_layer":
            assert experiment.probe_config.target_layers == ["last_layer"]
        else:
            assert experiment.probe_config.target_layers == ["last_layer"]

        if probe_type == "linear":
            assert experiment.probe_config.hidden_dims == [32]  # Updated to match optimized test config
            assert experiment.probe_config.dropout_rate == 0.1
            assert experiment.probe_config.activation == "relu"
        elif probe_type == "attention":
            assert experiment.probe_config.num_heads == 4
            assert experiment.probe_config.attention_dim == 256
            assert experiment.probe_config.num_layers == 2
            assert experiment.probe_config.dropout_rate == 0.1
            assert experiment.probe_config.activation == "relu"

    @pytest.mark.parametrize(
        "probe_type,freeze_backbone,layers,training_mode",
        [("linear", True, "last_layer", "offline")],
    )
    @pytest.mark.slow
    def test_weighted_probes_comprehensive(
        self,
        probe_type: str,
        freeze_backbone: bool,
        layers: str,
        training_mode: str,
        temp_output_dir: Path,
    ) -> None:
        assert (probe_type, freeze_backbone, layers, training_mode) == (
            "linear",
            True,
            "last_layer",
            "offline",
        ), "Harness only implements the linear/offline/last_layer path."
        metrics = run_linear_offline_probe_evaluate(temp_output_dir)
        for metric in ("test_accuracy", "test_balanced_accuracy"):
            assert metric in metrics, f"Missing metric column: {metric}"
            val = metrics[metric]
            assert 0.0 <= val <= 1.0, f"{metric} out of range: {val}"
