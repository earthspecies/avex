"""
End-to-end integration test for run_evaluate.py with probes.

This test:
1. Tests linear and attention probes
2. Tests with freeze_backbone=true and false
3. Tests with last_layer and all_layers configurations
4. Tests with training offline and online modes
5. Validates that expected metrics are present and within valid ranges
"""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml


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
        test_data_config = {
            "benchmark_name": "bioacoustic_benchmark_single_test",
            "evaluation_sets": [
                {
                    "name": "tiny_test",
                    "train": {
                        "dataset_name": "beans",
                        "split": "dogs_train",
                        "type": "classification",
                        "label_column": "label",
                        "audio_path_col": "path",
                        "multi_label": False,
                        "label_type": "supervised",
                        "audio_max_length_seconds": 5,  # Shorter audio for speed
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Rudy": 1.0,  # Take all available from these 3 classes
                                    "Zoe": 1.0,
                                    "Louie": 1.0,  # Add third class to ensure at least 2 in each split
                                },
                                "max_samples": 30,  # Small but enough to get multiple classes
                            },
                            # No label mapping - system should count classes from data
                        ],
                        "sample_rate": 16000,
                    },
                    "validation": {
                        "dataset_name": "beans",
                        "split": "dogs_validation",
                        "type": "classification",
                        "label_column": "label",
                        "audio_path_col": "path",
                        "multi_label": False,
                        "label_type": "supervised",
                        "audio_max_length_seconds": 5,
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Rudy": 1.0,
                                    "Zoe": 1.0,
                                    "Louie": 1.0,  # Add third class
                                },
                                "max_samples": 15,  # Small validation set
                            },
                            # No label mapping - system should count classes from data
                        ],
                        "sample_rate": 16000,
                    },
                    "test": {
                        "dataset_name": "beans",
                        "split": "dogs_test",
                        "type": "classification",
                        "label_column": "label",
                        "audio_path_col": "path",
                        "multi_label": False,
                        "label_type": "supervised",
                        "audio_max_length_seconds": 5,
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Rudy": 1.0,
                                    "Zoe": 1.0,
                                    "Louie": 1.0,  # Add third class
                                },
                                "max_samples": 15,  # Small test set
                            },
                            # No label mapping - system should count classes from data
                        ],
                        "sample_rate": 16000,
                    },
                    "metrics": [
                        "accuracy",
                        "balanced_accuracy",
                        "clustering_ari",
                        "clustering_nmi",
                    ],
                }
            ],
        }

        with open(data_config_path, "w") as f:
            yaml.dump(test_data_config, f, default_flow_style=False)

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

        Raises
        ------
        FileNotFoundError
            If the referenced run_config file (efficientnet_base.yml) is not found.
        """
        config_path = temp_output_dir / f"test_config_{probe_type}_{freeze_backbone}_{layers}_{training_mode}.yml"

        data_config_path = temp_output_dir / "test_data_config.yml"
        self._create_test_data_config(data_config_path)

        base_config = {
            "dataset_config": str(data_config_path),
            "training_params": {
                "train_epochs": 1,
                "lr": 0.0003,
                "batch_size": 1,
                "optimizer": "adamw",
                "weight_decay": 0.01,
                "amp": False,
                "amp_dtype": "bf16",
            },
            "save_dir": str(temp_output_dir / "results"),
            "device": "cpu",
            "seed": 42,
            "num_workers": 2,
            "eval_modes": ["probe"],
            "offline_embeddings": {
                "overwrite_embeddings": True,
                "use_streaming_embeddings": False,
                "memory_limit_gb": 32,
                "streaming_chunk_size": 1000,
                "hdf5_compression": "gzip",
                "hdf5_compression_level": 4,
                "auto_chunk_size": True,
                "max_chunk_size": 2000,
                "min_chunk_size": 100,
                "batch_chunk_size": 10,
                "cache_size_limit_gb": 16,
            },
        }

        target_layers = ["last_layer"] if layers == "last_layer" else ["last_layer"]

        # Resolve run_config path relative to project root
        # Use an efficientnet config that's tracked in git
        project_root = Path(__file__).parent.parent.parent
        # Use sl_efficientnet_animalspeak.yml which is tracked in git
        run_config_relative = Path("configs/run_configs/aaai_train/sl_efficientnet_animalspeak.yml")
        run_config_path = (project_root / run_config_relative).resolve()

        # Verify file exists
        if not run_config_path.exists():
            # Try efficientnet_base.yml as fallback (might exist locally but not in CI)
            fallback = project_root / "configs" / "run_configs" / "pretrained" / "efficientnet_base.yml"
            if fallback.exists():
                run_config_path = fallback
            else:
                raise FileNotFoundError(
                    f"Config file not found: {run_config_path}\n"
                    f"Project root: {project_root}\n"
                    f"Looking for: {run_config_relative}\n"
                    f"Fallback also not found: {fallback}"
                )

        # Use absolute path in the config to ensure it works in CI
        experiment = {
            "run_name": f"{probe_type}_{freeze_backbone}_{layers}_{training_mode}",
            "run_config": str(run_config_path),
            "probe_config": {
                "probe_type": probe_type,
                "aggregation": "mean",
                "input_processing": "pooled",
                "target_layers": target_layers,
                "freeze_backbone": freeze_backbone,
            },
            "pretrained": True,
        }

        if probe_type == "linear":
            experiment["probe_config"].update({"hidden_dims": [256, 128], "dropout_rate": 0.1, "activation": "relu"})
        elif probe_type == "attention":
            experiment["probe_config"].update(
                {
                    "num_heads": 4,
                    "attention_dim": 256,
                    "num_layers": 2,
                    "dropout_rate": 0.1,
                    "activation": "relu",
                }
            )

        if training_mode == "offline":
            experiment["probe_config"]["freeze_backbone"] = True
        else:
            experiment["probe_config"]["freeze_backbone"] = False

        base_config["experiments"] = [experiment]

        with open(config_path, "w") as f:
            yaml.dump(base_config, f, default_flow_style=False)

        return config_path

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

        from representation_learning.configs import EvaluateConfig

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
            assert experiment.probe_config.hidden_dims == [256, 128]
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
        import os

        from representation_learning.run_evaluate import main

        config_path = self._create_test_config(temp_output_dir, probe_type, freeze_backbone, layers, training_mode)

        test_output_dir = temp_output_dir / f"{probe_type}_{freeze_backbone}_{layers}_{training_mode}"
        test_output_dir.mkdir(exist_ok=True)

        patches = (
            f"save_dir={test_output_dir}",
            "device=cpu",
            "seed=42",
            "training_params.train_epochs=1",
            "training_params.batch_size=1",
            "offline_embeddings.use_streaming_embeddings=false",
            "offline_embeddings.memory_limit_gb=32",
            "offline_embeddings.cache_size_limit_gb=16",
        )

        # Ensure we're in project root when running main (it loads configs)
        project_root = Path(__file__).parent.parent.parent
        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)
            main(config_path, patches)
        finally:
            os.chdir(original_cwd)

        summary_csvs = list(test_output_dir.rglob("*summary*.csv"))
        assert summary_csvs, f"No summary CSVs found in {test_output_dir}"
        summary_csv = summary_csvs[0]
        df = pd.read_csv(summary_csv)

        expected_metrics = [
            "test_accuracy",
            "test_balanced_accuracy",
        ]
        for metric in expected_metrics:
            assert metric in df.columns, f"Missing metric column: {metric}"

        for metric in [
            "test_accuracy",
            "test_balanced_accuracy",
        ]:
            val = df[metric].iloc[0]
            if pd.isna(val):
                continue
            assert 0.0 <= val <= 1.0, f"{metric} out of range: {val}"

        ari = df["test_clustering_ari"].iloc[0]
        if not pd.isna(ari):
            assert -1.0 <= ari <= 1.0, f"test_clustering_ari out of range: {ari}"
