from __future__ import annotations

# Ensure HF Hub avoids Xet before any transformers import occurs
import os

os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from representation_learning.configs import RunConfig
from representation_learning.data.audio_utils import pad_or_window
from representation_learning.data.dataset import build_dataloaders
from representation_learning.models.get_model import get_model


class MockDataset(Dataset[dict[str, Any]]):
    """Deprecated: retained for reference; unused in real-data path."""

    def __init__(self, n_samples: int = 5) -> None:
        self.n_samples = n_samples
        self.text_labels = [
            "A bird singing in the forest",
            "A dog barking loudly",
            "A cat meowing softly",
            "A cow mooing in the field",
            "A horse neighing",
        ]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_length = 160000
        wav_np = torch.randn(audio_length).numpy()
        return {
            "raw_wav": wav_np,
            "text_label": self.text_labels[idx % len(self.text_labels)],
        }


@torch.no_grad()
def test_clip_mini_inference() -> None:
    """Mini smoke-test that runs CLIP on 5 random AnimalSpeak validation clips.

    The test loads the *pre-trained* CLIP checkpoint specified in
    ``configs/run_configs/clip_base.yml`` and verifies that an audio/text
    similarity score can be computed end-to-end without errors.

    Raises
    ------
    AssertionError
        If no CLIP run-config is found or if inference fails.
    """
    device = torch.device("cpu")  # keep CI lightweight

    # ------------------------------------------------------------------
    # Load run-config & model (pre-trained weights)
    # ------------------------------------------------------------------
    cfg_candidates = [
        Path("configs/run_configs/clip_base.yml"),
        Path("configs/run_configs/aaai_train/clap_efficientnet_captions.yml"),
    ]
    cfg_path = next((p for p in cfg_candidates if p.exists()), None)
    if cfg_path is None:
        raise AssertionError(
            "No CLIP run-config found; expected clip_base.yml or "
            "aaai_train/clap_efficientnet_captions.yml"
        )
    run_cfg = RunConfig.from_sources(yaml_file=cfg_path, cli_args=())

    # Ensure no training-time augmentations, just deterministic centre crop
    run_cfg.augmentations = []
    run_cfg.model_spec.audio_config.window_selection = "center"

    run_cfg.model_spec.device = "cpu"
    model = get_model(run_cfg.model_spec, num_classes=1)
    model.eval()
    model.to(device)

    # --------------------------------------------------------------
    # Load checkpoint weights if the run-config specifies one
    # --------------------------------------------------------------
    ckpt_path = getattr(run_cfg, "resume_from_checkpoint", None)
    if ckpt_path:
        ckpt_file = Path(ckpt_path)
        if ckpt_file.exists():
            state = torch.load(ckpt_file, map_location=device)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                print(f"Loaded checkpoint weights from {ckpt_file}")
        else:
            print(
                f"Warning: checkpoint path {ckpt_file} not found – using base weights"
            )

    # ------------------------------------------------------------------
    # Choose between mock data (default) and real data via env toggle
    # ------------------------------------------------------------------
    use_real = os.environ.get("CLIP_TEST_USE_REAL", "0") == "1"
    if use_real:
        # Use dataset config from run config; rely on esp_data defaults
        data_cfg = run_cfg.dataset_config
        if hasattr(data_cfg, "transformations"):
            data_cfg.transformations = None
        for lst_name in ("train_datasets", "val_datasets", "test_datasets"):
            lst = getattr(data_cfg, lst_name, None)
            if lst:
                for ds in lst:
                    ds.data_root = ""

        try:
            _, val_dl, _ = build_dataloaders(
                run_cfg,
                data_config=data_cfg,
                device="cpu",
                is_evaluation_context=True,
            )
        except FileNotFoundError:
            pytest.skip("Real audio files not found under esp_data defaults")

        try:
            batch = next(iter(val_dl))
        except StopIteration:
            pytest.skip("Validation dataloader is empty under esp_data defaults")

        if "text_label" not in batch:
            if "label" not in batch:
                pytest.skip("No text_label or label in batch for CLIP test")
            labels = batch["label"].tolist()
            texts = [f"class_{int(label)}" for label in labels]
        else:
            texts = list(batch["text_label"])  # type: ignore[index]

        B = batch["raw_wav"].shape[0]
        k = min(5, B)
        wav_batch = batch["raw_wav"][:k].to(device)
        mask_batch = batch.get("padding_mask")
        if mask_batch is None:
            mask_batch = torch.ones_like(wav_batch, dtype=torch.bool)
        else:
            mask_batch = mask_batch[:k].to(device)
        texts = texts[:k]
    else:
        # Mock dataset path for fast, reliable CI
        class _MockDataset(torch.utils.data.Dataset):
            def __init__(self, n: int = 5) -> None:
                self.n = n
                self.labels = [
                    "A bird singing in the forest",
                    "A dog barking loudly",
                    "A cat meowing softly",
                    "A cow mooing in the field",
                    "A horse neighing",
                ]

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict[str, Any]:
                audio_len = int(
                    run_cfg.model_spec.audio_config.target_length_seconds
                    * run_cfg.model_spec.audio_config.sample_rate
                )
                wav = torch.randn(audio_len)
                return {
                    "raw_wav": wav,
                    "text_label": self.labels[idx % len(self.labels)],
                }

        ds = _MockDataset(5)
        waves, masks, texts = [], [], []
        target_len = int(
            run_cfg.model_spec.audio_config.target_length_seconds
            * run_cfg.model_spec.audio_config.sample_rate
        )
        for i in range(len(ds)):
            item = ds[i]
            wav_t, mask_t = pad_or_window(item["raw_wav"], target_len, "center")
            waves.append(wav_t)
            masks.append(mask_t)
            texts.append(item["text_label"])
        wav_batch = torch.stack(waves).to(device)
        mask_batch = torch.stack(masks).to(device)

    # Ensure correct target length if not already collated
    target_len = int(
        run_cfg.model_spec.audio_config.target_length_seconds
        * run_cfg.model_spec.audio_config.sample_rate
    )
    if wav_batch.shape[1] != target_len:
        fixed = []
        fixed_mask = []
        for i in range(k):
            w, m = pad_or_window(wav_batch[i].cpu(), target_len, "center")
            fixed.append(w)
            fixed_mask.append(m)
        wav_batch = torch.stack(fixed).to(device)
        mask_batch = torch.stack(fixed_mask).to(device)

    # --------------------------------------------------------------
    # Forward pass through CLIP
    # --------------------------------------------------------------
    audio_emb, text_emb, logit_scale = model(wav_batch, texts, mask_batch)
    sim_matrix = (audio_emb @ text_emb.T * logit_scale).cpu()

    # Pretty-print matrix --------------------------------------------------
    print("\nSimilarity matrix (audio rows × text columns):")
    header = "      " + "  ".join([f"T{j}" for j in range(len(texts))])
    print(header)
    for i in range(len(texts)):
        row_vals = "  ".join(f"{sim_matrix[i, j]:6.2f}" for j in range(len(texts)))
        print(f"A{i} | {row_vals}")

    diag_vals = torch.diag(sim_matrix)
    off_diag_vals = sim_matrix[~torch.eye(len(texts), dtype=torch.bool)]

    print(
        "\nDiagonal similarities (paired samples):",
        [f"{v:.2f}" for v in diag_vals],
    )
    print("Mean diagonal:", diag_vals.mean().item())
    print("Mean off-diag :", off_diag_vals.mean().item())


if __name__ == "__main__":
    test_clip_mini_inference()
