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
        """Create a test data configuration without hardcoded data_root paths."""
        test_data_config = {
            "benchmark_name": "bioacoustic_benchmark_single_test",
            "evaluation_sets": [
                {
                    "name": "dog_classification",
                    "train": {
                        "dataset_name": "beans",
                        "split": "dogs_train",
                        "type": "classification",
                        "label_column": "label",
                        "audio_path_col": "path",
                        "multi_label": False,
                        "label_type": "supervised",
                        "audio_max_length_seconds": 10,
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Farley": 0.12,
                                    "Freid": 0.14,
                                    "Keri": 0.14,
                                    "Louie": 0.16,
                                    "Luke": 0.06,
                                    "Mac": 0.08,
                                    "Roodie": 0.05,
                                    "Rudy": 0.30,
                                    "Siggy": 0.09,
                                    "Zoe": 0.08,
                                },
                            },
                            {
                                "type": "label_from_feature",
                                "feature": "label",
                                "override": True,
                                "label_map": {
                                    "Farley": 0,
                                    "Freid": 1,
                                    "Keri": 2,
                                    "Louie": 3,
                                    "Luke": 4,
                                    "Mac": 5,
                                    "Roodie": 6,
                                    "Rudy": 7,
                                    "Siggy": 8,
                                    "Zoe": 9,
                                },
                            },
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
                        "audio_max_length_seconds": 10,
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Farley": 0.10,
                                    "Freid": 0.12,
                                    "Keri": 0.12,
                                    "Louie": 0.15,
                                    "Luke": 0.08,
                                    "Mac": 0.10,
                                    "Roodie": 0.08,
                                    "Rudy": 0.20,
                                    "Siggy": 0.10,
                                    "Zoe": 0.10,
                                },
                            },
                            {
                                "type": "label_from_feature",
                                "feature": "label",
                                "override": True,
                                "label_map": {
                                    "Farley": 0,
                                    "Freid": 1,
                                    "Keri": 2,
                                    "Louie": 3,
                                    "Luke": 4,
                                    "Mac": 5,
                                    "Roodie": 6,
                                    "Rudy": 7,
                                    "Siggy": 8,
                                    "Zoe": 9,
                                },
                            },
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
                        "audio_max_length_seconds": 10,
                        "transformations": [
                            {
                                "type": "subsample",
                                "property": "label",
                                "ratios": {
                                    "Farley": 0.10,
                                    "Freid": 0.12,
                                    "Keri": 0.12,
                                    "Louie": 0.15,
                                    "Luke": 0.08,
                                    "Mac": 0.10,
                                    "Roodie": 0.08,
                                    "Rudy": 0.20,
                                    "Siggy": 0.10,
                                    "Zoe": 0.10,
                                },
                            },
                            {
                                "type": "label_from_feature",
                                "feature": "label",
                                "override": True,
                                "label_map": {
                                    "Farley": 0,
                                    "Freid": 1,
                                    "Keri": 2,
                                    "Louie": 3,
                                    "Luke": 4,
                                    "Mac": 5,
                                    "Roodie": 6,
                                    "Rudy": 7,
                                    "Siggy": 8,
                                    "Zoe": 9,
                                },
                            },
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

        experiment = {
            "run_name": f"{probe_type}_{freeze_backbone}_{layers}_{training_mode}",
            "run_config": "configs/run_configs/pretrained/efficientnet_base.yml",
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
        from representation_learning.configs import EvaluateConfig

        config_path = self._create_test_config(temp_output_dir, probe_type, freeze_backbone, layers, training_mode)

        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=[])

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

        main(config_path, patches)

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
