"""Zero-shot CLAP evaluation on the BEANS CBI test split.

For each cbi_test clip, compare its audio embedding against the text embeddings
of all class names; the predicted class is the argmax cosine similarity. Top-1
and top-5 accuracy are reported.

CBI labels are 6-letter eBird banding codes (e.g. ``aldfly``). Those tokens
mean nothing to RoBERTa, so we expand them to species names by joining the cbi
JSONL files with the Xeno-canto metadata on the recording id (cbi
``file_name="XC137570.wav"`` → XC ``xc_id=137570``).

The text prompt for class C is built from the most-frequent ``species_common``
and ``canonical_name`` strings across all cbi training rows for that code. We
ensemble two prompt templates (common name and scientific name) and average
their embeddings before scoring.
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_cbi")
logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------- #
def load_clap(checkpoint_path: str, run_dir_config: str, device: torch.device):
    from avex.configs import RunConfig
    from avex.models.get_model import get_model

    with open(run_dir_config) as f:
        raw = yaml.safe_load(f)
    if "model_spec" in raw and isinstance(raw["model_spec"], dict):
        raw["model_spec"]["device"] = "cuda" if device.type == "cuda" else "cpu"
    cfg = RunConfig(**raw)
    cfg.model_spec.device = "cuda" if device.type == "cuda" else "cpu"
    cfg.model_spec.audio_encoder_init_from = None  # checkpoint will overwrite

    model = get_model(cfg.model_spec, num_classes=0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded checkpoint epoch=%s (missing=%d, unexpected=%d)",
                ckpt.get("epoch"), len(missing), len(unexpected))
    model.eval()
    return model, cfg


def build_label_text_map(cbi_train_path: str, xc_csv_path: str) -> dict:
    """code (e.g. 'aldfly') -> {'species_common': str, 'canonical_name': str}."""
    import polars as pl

    logger.info("Loading XC metadata for code -> name mapping...")
    t0 = time.time()
    xc = pl.read_csv(xc_csv_path, infer_schema_length=5000, ignore_errors=True)\
           .select(["xc_id", "species_common", "canonical_name"])\
           .with_columns(pl.col("xc_id").cast(pl.Int64, strict=False))
    logger.info("XC rows: %d (%.1fs)", len(xc), time.time() - t0)

    # cbi_train: pull file_name -> XC id
    rows = []
    from esp_data.io.filesystem import filesystem_from_path
    fs = filesystem_from_path(cbi_train_path)
    with fs.open(cbi_train_path, "rb") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("file_name")
        .str.replace_all(r"^XC", "", literal=False)
        .str.replace_all(r"\.\w+$", "", literal=False)
        .cast(pl.Int64, strict=False)
        .alias("xc_id")
    )

    joined = df.join(xc, on="xc_id", how="left")
    # most-common species_common per label
    mapping: dict[str, dict] = {}
    for label_grp in joined.group_by("label"):
        code = label_grp[0][0] if isinstance(label_grp[0], tuple) else label_grp[0]
        sub = label_grp[1]
        sc = sub.drop_nulls("species_common")["species_common"].mode()
        cn = sub.drop_nulls("canonical_name")["canonical_name"].mode()
        mapping[code] = {
            "species_common": (sc[0] if len(sc) else None),
            "canonical_name": (cn[0] if len(cn) else None),
        }
    n_with_common = sum(1 for v in mapping.values() if v["species_common"])
    logger.info("Built code->name map: %d classes, %d with species_common",
                len(mapping), n_with_common)
    return mapping


# ----------------------------------------------------------------------- #
# Audio
# ----------------------------------------------------------------------- #
def load_audio(local_path: str, target_sr: int, target_seconds: int) -> np.ndarray:
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio
    import librosa

    full_path = anypath("gs://esp-ml-datasets/beans/v0.1.0/raw/") / local_path
    audio, sr = read_audio(full_path)
    audio = audio.astype(np.float32)
    audio = audio_stereo_to_mono(audio, mono_method="average")
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr,
                                 res_type="kaiser_best")
    target_len = target_sr * target_seconds
    if audio.shape[0] >= target_len:
        start = (audio.shape[0] - target_len) // 2
        audio = audio[start: start + target_len]
    else:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    return audio


class CBITestDataset(torch.utils.data.Dataset):
    def __init__(self, samples, target_sr, target_seconds):
        self.samples = samples
        self.target_sr = target_sr
        self.target_seconds = target_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        r = self.samples[idx]
        try:
            wav = load_audio(r["local_path"], self.target_sr, self.target_seconds)
        except Exception as e:
            # Mark broken samples with NaN; we'll skip them downstream
            logger.warning("Failed to load %s: %s", r["local_path"], e)
            wav = np.zeros(self.target_sr * self.target_seconds, dtype=np.float32)
            wav[0] = float("nan")
        return torch.from_numpy(wav), idx


# ----------------------------------------------------------------------- #
# Encoding
# ----------------------------------------------------------------------- #
def encode_text_prototypes(model, codes, name_map, device):
    """Build (n_classes, D) prototype embeddings by averaging common+sci prompts."""
    common_prompts = []
    sci_prompts = []
    for code in codes:
        m = name_map.get(code, {})
        common = m.get("species_common")
        sci = m.get("canonical_name")
        # Fall back gracefully if no name was found
        common_prompts.append(common if common else code)
        sci_prompts.append(sci if sci else code)

    with torch.no_grad():
        e_common = model.encode_text(common_prompts)
        e_sci = model.encode_text(sci_prompts)
    e = F.normalize(e_common + e_sci, dim=-1)
    return e


def encode_audios(model, samples, target_sr, target_seconds, device,
                  batch_size: int, num_workers: int):
    import multiprocessing as mp

    ds = CBITestDataset(samples, target_sr, target_seconds)
    # gcsfs (and other async fsspec backends) is not fork-safe; force spawn so
    # each worker gets a fresh asyncio loop / GCS client.
    ctx = mp.get_context("spawn") if num_workers > 0 else None
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        multiprocessing_context=ctx,
        persistent_workers=(num_workers > 0),
    )
    n = len(samples)
    all_emb = torch.empty((n, 512), dtype=torch.float32)
    valid = np.zeros(n, dtype=bool)
    seen = 0
    t0 = time.time()
    for wav, idxs in dl:
        wav = wav.to(device, non_blocking=True)
        nan_mask = torch.isnan(wav).any(dim=1)
        wav = torch.where(nan_mask.unsqueeze(1), torch.zeros_like(wav), wav)
        pad_mask = torch.zeros_like(wav, dtype=torch.bool)
        with torch.no_grad():
            emb = model.encode_audio(wav, pad_mask)
        emb = emb.float().cpu()
        for j, i in enumerate(idxs.tolist()):
            if not nan_mask[j]:
                all_emb[i] = emb[j]
                valid[i] = True
        seen += len(idxs)
        if seen % (batch_size * 20) == 0 or seen == n:
            logger.info("Encoded %d / %d (%.1f clips/s)", seen, n,
                        seen / max(time.time() - t0, 1e-6))
    return all_emb, valid


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #
def find_latest_checkpoint(runs_root: str) -> tuple[str, str]:
    """Pick the latest checkpoint under runs_root.

    Strategy:
      1. Find the run directory whose ``best_model.pt`` is most recently modified.
      2. Inside that run directory, prefer the highest-epoch
         ``checkpoint_epoch_NNN.pt`` over ``best_model.pt`` (so we evaluate the
         freshest weights, not the best-val-epoch ones).
    """
    import glob
    import re

    bests = sorted(glob.glob(os.path.join(runs_root, "**", "best_model.pt"),
                             recursive=True), key=os.path.getmtime, reverse=True)
    if not bests:
        raise FileNotFoundError(f"No best_model.pt found under {runs_root}")
    run_dir = os.path.dirname(bests[0])

    epoch_re = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
    epoch_files = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint_epoch_*.pt")):
        m = epoch_re.search(os.path.basename(p))
        if m:
            epoch_files.append((int(m.group(1)), p))
    if epoch_files:
        epoch_files.sort()
        ckpt = epoch_files[-1][1]  # highest epoch number
    else:
        ckpt = bests[0]

    cfg = os.path.join(run_dir, "config.yml")
    return ckpt, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--runs-root", default="/mnt/home/avex/runs/clap")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="Optional path to write JSON metrics")
    args = p.parse_args()

    if args.checkpoint is None or args.config is None:
        ckpt, cfg = find_latest_checkpoint(args.runs_root)
        args.checkpoint = args.checkpoint or ckpt
        args.config = args.config or cfg
        logger.info("Using latest checkpoint: %s", args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model
    model, cfg = load_clap(args.checkpoint, args.config, device)
    target_sr = cfg.model_spec.audio_config.sample_rate
    target_seconds = cfg.model_spec.audio_config.target_length_seconds

    # Code -> species name mapping (from cbi_train rows joined to XC)
    name_map = build_label_text_map(
        "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_train.jsonl",
        "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/all_20260203.csv",
    )

    # Load full cbi_test
    from esp_data.io.filesystem import filesystem_from_path
    cbi_test_path = "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_test.jsonl"
    fs = filesystem_from_path(cbi_test_path)
    samples = []
    with fs.open(cbi_test_path, "rb") as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info("cbi_test samples: %d", len(samples))

    # Class set
    codes = sorted({s["label"] for s in samples})
    code_to_idx = {c: i for i, c in enumerate(codes)}
    logger.info("cbi classes: %d", len(codes))

    # Show a few mapped examples
    for c in codes[:5]:
        m = name_map.get(c, {})
        logger.info("  %s -> common=%r  sci=%r", c, m.get("species_common"),
                    m.get("canonical_name"))

    # Text prototypes (n_classes, D)
    logger.info("Encoding text prototypes...")
    proto = encode_text_prototypes(model, codes, name_map, device)

    # Audio embeddings (n, D)
    logger.info("Encoding audio for %d cbi_test samples...", len(samples))
    audio_emb, valid = encode_audios(
        model, samples, target_sr, target_seconds, device,
        args.batch_size, args.num_workers,
    )

    # Score
    scores = audio_emb.to(device) @ proto.T  # (n, n_classes)
    pred_top1 = scores.argmax(dim=1).cpu().numpy()
    top5 = scores.topk(5, dim=1).indices.cpu().numpy()

    true_idx = np.array([code_to_idx[s["label"]] for s in samples])
    keep = valid
    n = int(keep.sum())
    top1 = (pred_top1[keep] == true_idx[keep]).mean()
    top5_correct = np.array([
        true_idx[i] in top5[i] for i in range(len(samples)) if keep[i]
    ])
    top5_acc = top5_correct.mean()

    print()
    print("=" * 60)
    print("ZERO-SHOT CLAP EVAL — full BEANS CBI test")
    print("=" * 60)
    print(f"checkpoint:    {args.checkpoint}")
    print(f"epoch:         {cfg.training_params.train_epochs}")
    print(f"n_classes:     {len(codes)}")
    print(f"n_samples:     {n}  (skipped {len(samples) - n} due to load errors)")
    print(f"chance:        {1.0 / len(codes):.4f}")
    print(f"top-1 acc:     {top1:.4f}")
    print(f"top-5 acc:     {top5_acc:.4f}")
    print(f"top-1 lift:    {top1 * len(codes):.1f}x over chance")

    if args.output:
        out = {
            "checkpoint": args.checkpoint,
            "n_classes": len(codes),
            "n_samples": int(n),
            "top1": float(top1),
            "top5": float(top5_acc),
            "chance": 1.0 / len(codes),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
