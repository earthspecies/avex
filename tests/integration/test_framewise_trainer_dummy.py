import sys
import types
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

# -----------------------------------------------------------------------------
# 1.  Patch optional third-party dependencies that may be missing in CI
# -----------------------------------------------------------------------------
# bitsandbytes – optional optimiser backend (not required for CPU tests)
bnb_stub = types.ModuleType("bitsandbytes")
optim_stub = types.ModuleType("optim")
optim_stub.PagedAdamW8bit = torch.optim.AdamW  # fallback
bnb_stub.optim = optim_stub
sys.modules.setdefault("bitsandbytes", bnb_stub)

# google-cloud-storage – only used for GCS paths in ExperimentLogger
# We stub the *Client* class so that imports succeed even if the package
# is absent.
google_mod = types.ModuleType("google")
cloud_mod = types.ModuleType("google.cloud")
storage_mod = types.ModuleType("google.cloud.storage")
client_mod = types.ModuleType("google.cloud.storage.client")

class _DummyClient:  # pylint: disable=too-few-public-methods
    """Minimal stub for google.cloud.storage.client.Client"""

    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001, ANN002, D401
        pass

a = _DummyClient  # to appease flake8 unused var – we need at least one attr
client_mod.Client = _DummyClient

# Wire the nested module hierarchy
sys.modules.setdefault("google", google_mod)
sys.modules.setdefault("google.cloud", cloud_mod)
sys.modules.setdefault("google.cloud.storage", storage_mod)
sys.modules.setdefault("google.cloud.storage.client", client_mod)

google_mod.cloud = cloud_mod  # type: ignore[attr-defined]
cloud_mod.storage = storage_mod  # type: ignore[attr-defined]
storage_mod.client = client_mod  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# 2.  Imports from the project (done *after* stubs are in place)
# -----------------------------------------------------------------------------
from representation_learning.evaluation.finetune import (
    train_and_eval_framewise_probe,
)
from representation_learning.configs import AudioConfig, TrainingParams
from representation_learning.models.dummy_model import Model as DummyModel
from representation_learning.utils import ExperimentLogger

# -----------------------------------------------------------------------------
# 3.  Simple dummy *strong-detection* dataset
# -----------------------------------------------------------------------------

SR = 16000  # sample rate (Hz)
DURATION_SECS = 1  # each clip is 1 second -> 50 frames at 50 FPS
N_FRAMES = 50
N_CLASSES = 3


class _DummyStrongDetectionDataset(torch.utils.data.Dataset):
    """Returns random waveforms *plus* frame-level targets suitable for strong SED."""

    def __init__(self, n_samples: int) -> None:  # noqa: D401
        super().__init__()
        self.n_samples = n_samples
        self.num_samples = SR * DURATION_SECS

    def __len__(self) -> int:  # noqa: D401
        return self.n_samples

    def __getitem__(self, idx):  # noqa: D401, ANN001
        # 1. Raw waveform & (dummy) per-frame padding mask – here no padding
        wav = torch.randn(self.num_samples, dtype=torch.float32)
        padding_mask = torch.zeros(N_FRAMES, dtype=torch.bool)

        # 2. Frame-level multi-hot targets: (T, C)
        targets = torch.randint(0, 2, (N_FRAMES, N_CLASSES), dtype=torch.float32)

        return {
            "raw_wav": wav,
            "padding_mask": padding_mask,
            "frame_targets": targets,
        }


# Collater for stacking tensors ------------------------------------------------

def _collate_fn(batch):  # noqa: D401
    wavs = torch.stack([b["raw_wav"] for b in batch])
    masks = torch.stack([b["padding_mask"] for b in batch])
    targets = torch.stack([b["frame_targets"] for b in batch])
    return {"raw_wav": wavs, "padding_mask": masks, "frame_targets": targets}


# -----------------------------------------------------------------------------
# 4.  Monkey-patch the DummyModel.extract_embeddings to ignore extra kwargs &
#     yield fixed-size frame embeddings (B, T, D).
# -----------------------------------------------------------------------------

def _patched_extract_embeddings(self, x, layers, *, padding_mask=None, average_over_time=True, framewise_embeddings=False, **kwargs):  # noqa: ANN001, D401, E501
    batch_size = x.shape[0]
    if framewise_embeddings:
        emb = torch.randn(batch_size, N_FRAMES, self.embedding_dim, device=x.device)
        return [emb]
    # clip-level – unused in this test
    return torch.randn(batch_size, self.embedding_dim, device=x.device)


# Apply the patch *before* any model instantiation
import inspect  # noqa: E402  pylint: disable=wrong-import-position

_original_extract = DummyModel.extract_embeddings  # keep reference for safety
DummyModel.extract_embeddings = _patched_extract_embeddings  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# 5.  The actual pytest
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("train_samples,val_samples,test_samples", [(4, 2, 2)])
def test_framewise_trainer_runs(tmp_path: Path, train_samples: int, val_samples: int, test_samples: int):  # noqa: D401, E501
    """End-to-end sanity test for *framewise* linear probe training.

    This verifies that the *FineTuneTrainer* + *FramewiseLinearProbe* stack
    can execute a forward/backward pass on CPU without crashing.
    """

    device = torch.device("cpu")

    # ---------------------------- datasets & loaders ------------------------ #
    train_ds = _DummyStrongDetectionDataset(train_samples)
    val_ds = _DummyStrongDetectionDataset(val_samples)
    test_ds = _DummyStrongDetectionDataset(test_samples)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=_collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=_collate_fn)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=_collate_fn)

    # ---------------------------- dummy backbone --------------------------- #
    audio_cfg = AudioConfig(sample_rate=SR, target_length_seconds=DURATION_SECS)
    backbone = DummyModel(num_classes=N_CLASSES, device="cpu", audio_config=audio_cfg, return_features_only=True)

    # ---------------------------- minimal config --------------------------- #
    training_params = TrainingParams(
        train_epochs=1,
        lr=1e-3,
        batch_size=2,
        optimizer="adam",
        weight_decay=0.0,
        amp=False,
        amp_dtype="bf16",
    )
    eval_cfg = SimpleNamespace(training_params=training_params, save_dir=str(tmp_path))

    exp_logger = ExperimentLogger(backend="none", handle=None)

    # ---------------------------- run probe -------------------------------- #
    layer_names = ["dummy"]  # ignored by DummyModel
    metric_names = ["f1_strong"]

    train_metrics, val_metrics, test_metrics = train_and_eval_framewise_probe(
        train_dl,
        val_dl,
        test_dl,
        backbone,
        N_CLASSES,
        layer_names,
        eval_cfg,
        device,
        exp_logger,
        metric_names,
    )

    # ---------------------------- assertions -------------------------------- #
    assert "f1_strong" in test_metrics, "Metric key missing in probe output."
    assert 0.0 <= test_metrics["f1_strong"] <= 1.0, "F1 score out of bounds."

    # Clean-up: restore original method to avoid side-effects on other tests
    DummyModel.extract_embeddings = _original_extract  # type: ignore[assignment] 