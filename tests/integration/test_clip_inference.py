import random
from pathlib import Path

import torch

from representation_learning.configs import load_config
from representation_learning.data.audio_utils import pad_or_window
from representation_learning.data.dataset import get_dataset_dummy
from representation_learning.models.get_model import get_model


@torch.no_grad()
def test_clip_mini_inference():
    """Mini smoke-test that runs CLIP on 5 random AnimalSpeak validation clips.

    The test loads the *pre-trained* CLIP checkpoint specified in
    ``configs/run_configs/clip_base.yml`` and verifies that an audio/text
    similarity score can be computed end-to-end without errors.
    """
    device = torch.device("cpu")  # keep CI lightweight

    # ------------------------------------------------------------------
    # Load run-config & model (pre-trained weights)
    # ------------------------------------------------------------------
    cfg_path = Path("configs/run_configs/clip_base.yml")
    run_cfg = load_config(cfg_path, config_type="run")

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
            print(f"Warning: checkpoint path {ckpt_file} not found – using base weights")

    # ------------------------------------------------------------------
    # Load AnimalSpeak validation split
    # ------------------------------------------------------------------
    data_cfg = load_config("configs/data_configs/data_base.yml", config_type="data")
    ds_val = get_dataset_dummy(data_cfg, split="valid")

    assert len(ds_val) > 0, "Dataset appears empty – check CSV path/GCS creds."

    sample_indices = random.sample(range(len(ds_val)), k=5)

    # --------------------------------------------------------------
    # Build batch tensors
    # --------------------------------------------------------------
    waves, masks, texts = [], [], []
    target_len = int(
        run_cfg.model_spec.audio_config.target_length_seconds
        * run_cfg.model_spec.audio_config.sample_rate
    )

    for idx in sample_indices:
        item = ds_val[idx]
        wav_np = item["raw_wav"]
        wav_t, mask_t = pad_or_window(torch.tensor(wav_np), target_len, "center")
        waves.append(wav_t)
        masks.append(mask_t)
        texts.append(item["text_label"])

    wav_batch = torch.stack(waves).to(device)
    mask_batch = torch.stack(masks).to(device)

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

    print("\nDiagonal similarities (paired samples):", [f"{v:.2f}" for v in diag_vals])
    print("Mean diagonal:", diag_vals.mean().item())
    print("Mean off-diag :", off_diag_vals.mean().item())


if __name__ == "__main__":
    test_clip_mini_inference()
