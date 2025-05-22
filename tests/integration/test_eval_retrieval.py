from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from representation_learning.configs import (
    AudioConfig,
    EvaluateConfig,
    ExperimentConfig,
    ModelSpec,
    RunConfig,
    TrainingParams,
)
from representation_learning.run_evaluate import run_experiment


# --------------------------------------------------------------------- #
#  Tiny dummy dataset
# --------------------------------------------------------------------- #
class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    """Minimal dataset yielding random waveforms and binary labels."""

    def __init__(self, n_samples: int = 8) -> None:
        self.n: int = n_samples
        self.metadata: dict[str, Any] = {"label_map": {0: 0, 1: 1}}  # noqa: ANN401

    # ------------------------------------------------------------------ #
    #  PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # noqa: D401
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D401
        wav: torch.Tensor = torch.randn(160)  # 10 ms dummy audio @ 16 kHz
        label: torch.Tensor = torch.randint(0, 2, ()).long()
        return {"raw_wav": wav, "label": label}


# --------------------------------------------------------------------- #
#  Mocks
# --------------------------------------------------------------------- #
def _mock_build_dataloaders(
    *_: object, **__: object
) -> Tuple[DataLoader, DataLoader, DataLoader]:  # noqa: D401
    """Return train/val/test loaders over the same tiny dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        *train*, *val* and *test* data loaders (identical in this stub).
    """
    ds: _TinyDataset = _TinyDataset()
    dl: DataLoader = DataLoader(ds, batch_size=4, shuffle=False)
    return dl, dl, dl


class _DummyModel(torch.nn.Module):
    """Single-layer linear dummy model mimicking ModelBase API."""

    def __init__(self) -> None:
        super().__init__()
        self.fc: torch.nn.Linear = torch.nn.Linear(160, 8)

    # Forward pass ----------------------------------------------------- #
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # noqa: D401
        return self.fc(x)

    # Embedding extractor used by evaluation --------------------------- #
    def extract_embeddings(
        self, x: torch.Tensor | dict[str, torch.Tensor], layers: str
    ) -> torch.Tensor:  # noqa: D401
        if isinstance(x, dict):
            x = x["raw_wav"]
        return self.forward(x)


# --------------------------------------------------------------------- #
#  Test
# --------------------------------------------------------------------- #
def test_run_experiment_small(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: D401
    """Verify that `run_experiment` executes end-to-end on a minimal setup."""
    # Patch dataloader builders ------------------------------------------------
    from representation_learning.data import dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "build_dataloaders", _mock_build_dataloaders)

    import representation_learning.run_evaluate as reval_mod

    monkeypatch.setattr(reval_mod, "build_dataloaders", _mock_build_dataloaders)

    # Patch model factory ------------------------------------------------------
    from representation_learning.models import get_model as get_model_mod

    def _get_model_stub(*_: object, **__: object) -> _DummyModel:  # noqa: D401
        return _DummyModel()

    monkeypatch.setattr(get_model_mod, "get_model", _get_model_stub)
    monkeypatch.setattr(reval_mod, "get_model", _get_model_stub)

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

    # Patch `load_config` to return a stub RunConfig ---------------------------
    def _load_config_stub(path: str | Path, config_type: str = "run") -> Any:  # noqa: D401, ANN401
        if config_type == "run":
            return RunConfig(
                model_spec=ModelSpec(
                    name="dummy",
                    pretrained=False,
                    audio_config=AudioConfig(),
                ),
                training_params=train_params,
                dataset_config="dummy",
                output_dir="/tmp",
                loss_function="cross_entropy",
            )
        return path  # pragma: no cover

    monkeypatch.setattr(reval_mod, "load_config", _load_config_stub)

    # ------------------------------------------------------------------------- #
    #  Build EvaluateConfig and ExperimentConfig
    # ------------------------------------------------------------------------- #
    eval_cfg: EvaluateConfig = EvaluateConfig(
        experiments=[],  # filled below
        dataset_config="dummy.yml",
        save_dir="/tmp",
        training_params=train_params,
        device="cpu",
        seed=0,
        num_workers=0,
        frozen=False,
        eval_modes=["retrieval"],
    )

    exp_cfg: ExperimentConfig = ExperimentConfig(
        run_name="dummy",
        run_config="dummy_run.yml",
        pretrained=True,
        layers="last_layer",
    )
    eval_cfg.experiments.append(exp_cfg)  # type: ignore[arg-type]

    dataset_cfg: SimpleNamespace = SimpleNamespace(
        dataset_name="dummy_ds",
        metrics=["accuracy"],
        multi_label=False,
    )

    # ------------------------------------------------------------------------- #
    #  Execute experiment
    # ------------------------------------------------------------------------- #
    result: Any = run_experiment(
        eval_cfg,
        dataset_cfg,  # type: ignore[arg-type]
        exp_cfg,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        save_dir=Path("/tmp"),
    )

    # Ensure that something sensible is returned ------------------------------
    assert result is not None
