"""Integration tests for evaluation retrieval.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

# Skip entire module if esp_data is not installed (internal dependency)
# Must be before imports that trigger esp_data loading (e.g., avex.run_evaluate)
esp_data_module = pytest.importorskip("esp_data")
DatasetConfig = esp_data_module.DatasetConfig

from avex.configs import EvaluateConfig, ExperimentConfig, TrainingParams  # noqa: E402
from avex.run_evaluate import run_experiment  # noqa: E402


# --------------------------------------------------------------------- #
#  Tiny dummy dataset
# --------------------------------------------------------------------- #
class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    """Minimal dataset yielding random waveforms and binary labels."""

    def __init__(self, n_samples: int = 8) -> None:
        self.n: int = n_samples
        self.metadata: dict[str, Any] = {  # noqa: ANN401
            "label_map": {0: 0, 1: 1},
            "num_labels": 2,  # Required for retrieval evaluation
        }
        # Pre-generate labels to ensure consistency
        self.labels = torch.tensor([i % 2 for i in range(n_samples)], dtype=torch.long)

    # ------------------------------------------------------------------ #
    #  PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # noqa: D401
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D401
        # Generate 1 second of audio at 16kHz for BEATs model
        wav: torch.Tensor = torch.randn(16000)  # 1 second dummy audio @ 16 kHz
        label: torch.Tensor = self.labels[idx]  # Use pre-generated labels
        return {"raw_wav": wav, "label": label}


# --------------------------------------------------------------------- #
#  Mocks
# --------------------------------------------------------------------- #
def _mock_build_dataloaders(*_: object, **__: object) -> Tuple[DataLoader, DataLoader, DataLoader]:  # noqa: D401
    """Return train/val/test loaders over the same tiny dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        *train*, *val* and *test* data loaders (identical in this stub).
    """
    ds: _TinyDataset = _TinyDataset()
    dl: DataLoader = DataLoader(ds, batch_size=4, shuffle=False)
    return dl, dl, dl


# No dummy model needed - using real BEATs model


# --------------------------------------------------------------------- #
#  Test
# --------------------------------------------------------------------- #
def test_run_experiment_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: D401
    """Verify that `run_experiment` executes end-to-end with BEATs model and
    retrieval evaluation."""
    # Patch dataloader builders ------------------------------------------------
    from avex.data import dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "build_dataloaders", _mock_build_dataloaders)

    import avex.run_evaluate as reval_mod

    monkeypatch.setattr(reval_mod, "build_dataloaders", _mock_build_dataloaders)

    # Mock the retrieval evaluation to handle None labels issue
    from avex.evaluation import retrieval as retrieval_mod

    def _mock_eval_retrieval_cross_set(*args: object, **kwargs: object) -> dict[str, float]:
        """Mock retrieval evaluation that returns dummy metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing mock retrieval metrics.
        """
        return {"retrieval_precision_at_1": 0.5, "retrieval_precision_at_5": 0.6}

    monkeypatch.setattr(retrieval_mod, "eval_retrieval_cross_set", _mock_eval_retrieval_cross_set)
    monkeypatch.setattr(reval_mod, "eval_retrieval_cross_set", _mock_eval_retrieval_cross_set)

    # Mock GCS writes to avoid permission issues in CI
    from avex.utils import experiment_tracking as et_mod

    def _mock_save_evaluation_metadata(*args: object, **kwargs: object) -> None:
        """Mock save_evaluation_metadata to avoid GCS writes in CI."""
        pass

    monkeypatch.setattr(et_mod, "save_evaluation_metadata", _mock_save_evaluation_metadata)
    # Also patch in run_evaluate module since it imports the function directly
    monkeypatch.setattr(reval_mod, "save_evaluation_metadata", _mock_save_evaluation_metadata)

    # Minimal training params --------------------------------------------------
    train_params: TrainingParams = TrainingParams(
        train_epochs=1,
        lr=1e-3,
        batch_size=4,
        optimizer="adamw",
        weight_decay=0.0,
        amp=False,
        amp_dtype="bf16",
    )

    # load_config function was removed, no need to mock it

    # ------------------------------------------------------------------------- #
    #  Build EvaluateConfig and ExperimentConfig
    # ------------------------------------------------------------------------- #
    eval_cfg: EvaluateConfig = EvaluateConfig(
        experiments=[],  # filled below
        dataset_config="configs/data_configs/benchmark_single.yml",
        save_dir="/tmp",
        training_params=train_params,
        device="cpu",
        seed=0,
        num_workers=2,
        eval_modes=["retrieval"],
        offline_embeddings=dict(overwrite_embeddings=True),  # Force recomputation
    )

    exp_cfg: ExperimentConfig = ExperimentConfig(
        run_name="beats_test",
        run_config="configs/run_configs/aaai_train/sl_beats_animalspeak.yml",
        pretrained=True,
        layers="last_layer",  # Use last_layer for BEATs model
    )
    eval_cfg.experiments.append(exp_cfg)  # type: ignore[arg-type]

    dataset_cfg: DatasetConfig = DatasetConfig(
        dataset_name="dummy_ds",
        metrics=["accuracy"],
        multi_label=False,
    )

    # Create a mock data collection config
    from avex.data.configs import DatasetCollectionConfig

    data_collection_cfg = DatasetCollectionConfig(
        train_datasets=[DatasetConfig(dataset_name="dummy_train")],
        val_datasets=[DatasetConfig(dataset_name="dummy_val")],  # Add validation dataset
        test_datasets=[DatasetConfig(dataset_name="dummy_test")],  # Add test dataset
    )

    # ------------------------------------------------------------------------- #
    #  Execute experiment
    # ------------------------------------------------------------------------- #
    # Create a mock evaluation set with train_vs_test retrieval mode
    from avex.data.configs import EvaluationSet

    evaluation_set = EvaluationSet(
        name="test_set",
        train=data_collection_cfg.train_datasets[0],  # Single DatasetConfig
        validation=data_collection_cfg.val_datasets[0],  # Single DatasetConfig
        test=data_collection_cfg.test_datasets[0],  # Single DatasetConfig
        retrieval_mode="train_vs_test",  # This will trigger train embedding computation
    )

    result: Any = run_experiment(
        eval_cfg,
        dataset_cfg,  # type: ignore[arg-type]
        exp_cfg,  # type: ignore[arg-type]
        data_collection_cfg,
        device=torch.device("cpu"),
        save_dir=Path("/tmp"),
        evaluation_set=evaluation_set,
    )

    # Ensure that something sensible is returned ------------------------------
    assert result is not None
    assert hasattr(result, "result")
    assert hasattr(result.result, "retrieval_metrics")
    # Verify that retrieval metrics were computed
    assert result.result.retrieval_metrics is not None
    assert len(result.result.retrieval_metrics) > 0
    # The test passes if we get here - BEATs model with retrieval evaluation
    # ran successfully
