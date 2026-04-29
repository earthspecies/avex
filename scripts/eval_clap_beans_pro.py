"""Zero-shot CLAP evaluation on a BEANS-Pro multiple-choice description task.

Each item has:
- audio clip (referenced by ``audio_path_original_sample_rate``)
- 4 candidate acoustic descriptions labelled A/B/C/D, embedded in ``instruction_text``
- ground-truth letter in ``output``

Eval strategy: encode the audio, encode each of the 4 description strings, pick
the choice with highest cosine similarity between the audio and the description
embedding. Compare to ``output`` for accuracy.

Default split: ``crow-description`` (200 examples, carrion crow, 25 call types).
"""

import argparse
import json
import logging
import os
import re
import time
from typing import Optional

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
# Default per-split paths (mirrors the BeansPro dataset class)
# ----------------------------------------------------------------------- #
SPLIT_PATHS = {
    "crow-description": (
        "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/test.jsonl",
        "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/",
    ),
    "zebra-description": (
        "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/test.jsonl",
        "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/",
    ),
}


CHOICE_RE = re.compile(r"^([A-D]):\s*(.*)$")


def parse_choices(instruction_text: str) -> dict[str, str]:
    """Extract {'A': desc, 'B': desc, 'C': desc, 'D': desc} from the prompt."""
    choices: dict[str, str] = {}
    current_letter: Optional[str] = None
    current_lines: list[str] = []
    for raw in instruction_text.split("\n"):
        line = raw.rstrip()
        m = CHOICE_RE.match(line)
        if m:
            if current_letter is not None:
                choices[current_letter] = " ".join(current_lines).strip()
            current_letter = m.group(1)
            current_lines = [m.group(2)]
        elif line.startswith("Answer"):
            if current_letter is not None and current_letter not in choices:
                choices[current_letter] = " ".join(current_lines).strip()
            current_letter = None
            current_lines = []
        elif current_letter is not None and line.strip():
            current_lines.append(line.strip())
    if current_letter is not None and current_letter not in choices:
        choices[current_letter] = " ".join(current_lines).strip()
    return choices


# ----------------------------------------------------------------------- #
# Model loading
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
    return model, cfg, ckpt.get("epoch")


def find_latest_checkpoint(runs_root: str) -> tuple[str, str]:
    """Pick the freshest checkpoint under runs_root.

    Find the run dir whose ``best_model.pt`` is most recently modified, then
    inside that dir prefer the highest-epoch ``checkpoint_epoch_NNN.pt`` over
    ``best_model.pt`` so we evaluate the freshest weights.
    """
    import glob
    import re

    bests = sorted(glob.glob(os.path.join(runs_root, "**", "best_model.pt"),
                             recursive=True), key=os.path.getmtime, reverse=True)
    if not bests:
        raise FileNotFoundError(f"No best_model.pt under {runs_root}")
    run_dir = os.path.dirname(bests[0])

    epoch_re = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
    epoch_files = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint_epoch_*.pt")):
        m = epoch_re.search(os.path.basename(p))
        if m:
            epoch_files.append((int(m.group(1)), p))
    if epoch_files:
        epoch_files.sort()
        ckpt = epoch_files[-1][1]  # highest epoch
    else:
        ckpt = bests[0]

    cfg = os.path.join(run_dir, "config.yml")
    return ckpt, cfg


# ----------------------------------------------------------------------- #
# Audio loading
# ----------------------------------------------------------------------- #
def load_audio(local_path: str, data_root: str, target_sr: int,
               target_seconds: int) -> np.ndarray:
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio
    import librosa

    full_path = anypath(data_root) / local_path
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


class BeansProAudioDS(torch.utils.data.Dataset):
    def __init__(self, samples, data_root, target_sr, target_seconds):
        self.samples = samples
        self.data_root = data_root
        self.target_sr = target_sr
        self.target_seconds = target_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        r = self.samples[idx]
        rel = r["audio_path_original_sample_rate"]
        try:
            wav = load_audio(rel, self.data_root, self.target_sr, self.target_seconds)
        except Exception as e:
            logger.warning("Audio load failed for %s: %s", rel, e)
            wav = np.zeros(self.target_sr * self.target_seconds, dtype=np.float32)
            wav[0] = float("nan")
        return torch.from_numpy(wav), idx


# ----------------------------------------------------------------------- #
# Eval
# ----------------------------------------------------------------------- #
def encode_audios(model, samples, data_root, target_sr, target_seconds,
                  device, batch_size: int, num_workers: int) -> tuple[torch.Tensor, np.ndarray]:
    import multiprocessing as mp

    ds = BeansProAudioDS(samples, data_root, target_sr, target_seconds)
    ctx = mp.get_context("spawn") if num_workers > 0 else None
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        multiprocessing_context=ctx, persistent_workers=(num_workers > 0),
    )
    n = len(samples)
    embs = torch.empty((n, 512), dtype=torch.float32)
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
                embs[i] = emb[j]
                valid[i] = True
        seen += len(idxs)
        if seen % (batch_size * 20) == 0 or seen == n:
            logger.info("Encoded %d / %d (%.1f clips/s)", seen, n,
                        seen / max(time.time() - t0, 1e-6))
    return embs, valid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--runs-root", default="/mnt/home/avex/runs/clap")
    p.add_argument("--split", default="crow-description",
                   choices=sorted(SPLIT_PATHS.keys()))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only run on the first N items (for smoke testing)")
    p.add_argument("--output", default=None,
                   help="Path to write summary JSON metrics")
    p.add_argument("--per-item-output", default=None,
                   help="Path to write per-item JSONL with options, sims, prediction, gt")
    args = p.parse_args()

    if args.checkpoint is None or args.config is None:
        ckpt, cfg = find_latest_checkpoint(args.runs_root)
        args.checkpoint = args.checkpoint or ckpt
        args.config = args.config or cfg
        logger.info("Auto-selected checkpoint: %s", args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, cfg, ckpt_epoch = load_clap(args.checkpoint, args.config, device)
    target_sr = cfg.model_spec.audio_config.sample_rate
    target_seconds = cfg.model_spec.audio_config.target_length_seconds

    # Load JSONL
    jsonl_path, default_root = SPLIT_PATHS[args.split]
    from esp_data.io.filesystem import filesystem_from_path
    fs = filesystem_from_path(jsonl_path)
    samples = []
    with fs.open(jsonl_path, "rb") as f:
        for line in f:
            samples.append(json.loads(line))
    if args.limit > 0:
        samples = samples[: args.limit]
    logger.info("Loaded %d items from split=%s", len(samples), args.split)

    # Parse choices and ground truth
    correct_letters: list[str] = []
    choice_lists: list[list[str]] = []  # list of [optA, optB, optC, optD] per row
    skipped = 0
    for r in samples:
        chs = parse_choices(r.get("instruction_text") or r.get("instruction", ""))
        if any(letter not in chs for letter in "ABCD"):
            logger.warning("Item %s missing some choices: keys=%s", r.get("id"),
                           sorted(chs.keys()))
            skipped += 1
            chs = {ltr: chs.get(ltr, "") for ltr in "ABCD"}
        choice_lists.append([chs["A"], chs["B"], chs["C"], chs["D"]])
        correct_letters.append(r["output"].strip().upper())

    # Sample: show one item to confirm parsing
    logger.info("First item — correct=%s, choice A: %s",
                correct_letters[0], choice_lists[0][0][:120])

    # Encode all texts: 4 per row, flatten (N*4, D)
    flat_texts = [t for choices in choice_lists for t in choices]
    logger.info("Encoding %d candidate descriptions (4 per item)...", len(flat_texts))
    txt_embs = []
    txt_bs = 32
    with torch.no_grad():
        for s in range(0, len(flat_texts), txt_bs):
            chunk = flat_texts[s: s + txt_bs]
            txt_embs.append(model.encode_text(chunk).float().cpu())
    txt = torch.cat(txt_embs, dim=0).reshape(len(samples), 4, -1)  # (N, 4, D)

    # Encode audios
    logger.info("Encoding %d audio clips...", len(samples))
    aud, valid = encode_audios(model, samples, default_root, target_sr,
                               target_seconds, device, args.batch_size,
                               args.num_workers)

    # Score: cosine sim audio (N,D) vs text (N,4,D) — embeddings already normalised
    sims = (aud.unsqueeze(1) * txt).sum(dim=-1)  # (N, 4)
    pred_idx = sims.argmax(dim=1).numpy()
    letters = np.array(["A", "B", "C", "D"])
    pred_letters = letters[pred_idx]

    # Random/chance baseline = 0.25
    correct_arr = np.array(correct_letters)
    keep = valid & (np.array([(c in "ABCD") for c in correct_arr]))
    n = int(keep.sum())
    correct = (pred_letters[keep] == correct_arr[keep]).sum()
    acc = correct / n if n else float("nan")

    # Per-letter distribution sanity check
    pred_dist = {ltr: int((pred_letters[keep] == ltr).sum()) for ltr in "ABCD"}
    gt_dist = {ltr: int((correct_arr[keep] == ltr).sum()) for ltr in "ABCD"}

    print()
    print("=" * 70)
    print(f"BEANS-Pro / {args.split}  —  zero-shot CLAP MCQ")
    print("=" * 70)
    print(f"checkpoint:        {args.checkpoint}")
    print(f"checkpoint epoch:  {ckpt_epoch}")
    print(f"n items:           {n}  (skipped {len(samples) - n})")
    print(f"chance:            0.2500")
    print(f"top-1 accuracy:    {acc:.4f}  ({correct}/{n})")
    print(f"prediction dist:   {pred_dist}")
    print(f"ground-truth dist: {gt_dist}")

    if args.per_item_output:
        with open(args.per_item_output, "w") as f:
            for i, r in enumerate(samples):
                if not valid[i]:
                    continue
                row_sims = sims[i].tolist()  # length-4
                order = sorted(range(4), key=lambda k: -row_sims[k])
                ranking = [
                    {
                        "letter": "ABCD"[k],
                        "similarity": float(row_sims[k]),
                        "is_gt": ("ABCD"[k] == correct_letters[i]),
                        "option_text": choice_lists[i][k],
                    }
                    for k in order
                ]
                rec = {
                    "id": r.get("id"),
                    "file_name": r.get("file_name"),
                    "audio_path": r.get("audio_path_original_sample_rate"),
                    "ground_truth": correct_letters[i],
                    "predicted": str(pred_letters[i]),
                    "correct": bool(pred_letters[i] == correct_letters[i]),
                    "metadata": r.get("metadata"),
                    "ranking": ranking,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote per-item details to %s", args.per_item_output)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "epoch": ckpt_epoch,
                "split": args.split,
                "n_items": int(n),
                "accuracy": float(acc),
                "pred_dist": pred_dist,
                "gt_dist": gt_dist,
            }, f, indent=2)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
