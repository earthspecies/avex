"""Zero-shot BioLingual (HuggingFace) evaluation on BEANS-Pro multiple-choice.

Same task as eval_clap_beans_pro.py but uses davidrrobinson/BioLingual
(transformers ClapModel) instead of our locally trained CLAP. For comparison.
"""

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# Reuse the parser + paths from the local script.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clap_beans_pro import parse_choices, SPLIT_PATHS  # noqa: E402


def load_audio(local_path: str, data_root: str, target_sr: int) -> np.ndarray:
    """Load + (lightly) preprocess audio at the model's expected sample rate.

    BioLingual (ClapModel) expects 48 kHz mono. The HF processor will pad/truncate
    internally; we just hand it the raw waveform.
    """
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio
    import librosa

    full_path = anypath(data_root) / local_path
    audio, sr = read_audio(full_path)
    audio = audio.astype(np.float32)
    audio = audio_stereo_to_mono(audio, mono_method="average")
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr,
                                 res_type="kaiser_best")
    return audio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="crow-description",
                   choices=sorted(SPLIT_PATHS.keys()))
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model-id", default="davidrrobinson/BioLingual")
    p.add_argument("--output", default=None,
                   help="Path to write summary JSON metrics")
    p.add_argument("--per-item-output", default=None,
                   help="Path to write per-item JSONL with options, sims, prediction, gt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # The HF model is published as pytorch_model.bin (no safetensors). transformers
    # gates torch.load on torch>=2.6; we trust this checkpoint, so neutralize the
    # check (must patch BOTH import_utils and modeling_utils, since modeling_utils
    # imports the function by name into its own namespace at module load time).
    import transformers.utils.import_utils as _imp_utils
    import transformers.modeling_utils as _mod_utils
    _imp_utils.check_torch_load_is_safe = lambda *a, **k: None
    _mod_utils.check_torch_load_is_safe = lambda *a, **k: None
    from transformers import ClapModel, ClapProcessor

    logger.info("Loading %s ...", args.model_id)
    t0 = time.time()
    model = ClapModel.from_pretrained(args.model_id).to(device).eval()
    proc = ClapProcessor.from_pretrained(args.model_id)
    logger.info("Loaded in %.1fs (params=%d)", time.time() - t0,
                sum(p.numel() for p in model.parameters()))

    # Detect expected sample rate from feature extractor (default 48k for CLAP)
    target_sr = getattr(proc.feature_extractor, "sampling_rate", 48000)
    logger.info("Processor sampling_rate: %d", target_sr)

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
    logger.info("Loaded %d items from %s", len(samples), args.split)

    correct_letters = []
    choice_lists = []
    for r in samples:
        chs = parse_choices(r.get("instruction_text") or r.get("instruction", ""))
        choice_lists.append([chs.get(l, "") for l in "ABCD"])
        correct_letters.append(r["output"].strip().upper())
    logger.info("First item: correct=%s, A: %s", correct_letters[0],
                choice_lists[0][0][:100])

    # ---- Encode text (4 per item) ----
    flat_texts = [t for choices in choice_lists for t in choices]
    text_inputs = proc(text=flat_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=200).to(device)
    with torch.no_grad():
        txt = model.get_text_features(**text_inputs).float()
    txt = F.normalize(txt, dim=-1).cpu()
    txt = txt.reshape(len(samples), 4, -1)
    logger.info("Encoded %d candidate texts (shape %s)", len(flat_texts), tuple(txt.shape))

    # ---- Encode audio one at a time ----
    aud = torch.empty((len(samples), txt.shape[-1]), dtype=torch.float32)
    valid = np.zeros(len(samples), dtype=bool)
    t0 = time.time()
    for i, r in enumerate(samples):
        try:
            wav = load_audio(r["audio_path_original_sample_rate"], default_root, target_sr)
        except Exception as e:
            logger.warning("Audio load fail %s: %s", r["audio_path_original_sample_rate"], e)
            continue
        a_in = proc(audios=[wav], sampling_rate=target_sr,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            a_emb = model.get_audio_features(**a_in).float()
        a_emb = F.normalize(a_emb, dim=-1).cpu()
        aud[i] = a_emb[0]
        valid[i] = True
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            logger.info("Encoded %d/%d (%.1f clips/s)", i + 1, len(samples),
                        (i + 1) / max(time.time() - t0, 1e-6))

    # ---- Score ----
    sims = (aud.unsqueeze(1) * txt).sum(dim=-1)  # (N, 4)
    pred_idx = sims.argmax(dim=1).numpy()
    letters = np.array(["A", "B", "C", "D"])
    pred = letters[pred_idx]
    gt = np.array(correct_letters)
    keep = valid & np.array([(c in "ABCD") for c in gt])
    n = int(keep.sum())
    acc = (pred[keep] == gt[keep]).sum() / n if n else float("nan")

    pred_dist = {l: int((pred[keep] == l).sum()) for l in "ABCD"}
    gt_dist = {l: int((gt[keep] == l).sum()) for l in "ABCD"}

    print()
    print("=" * 70)
    print(f"BEANS-Pro / {args.split} — BioLingual ({args.model_id})")
    print("=" * 70)
    print(f"n items:           {n}")
    print(f"chance:            0.2500")
    print(f"top-1 accuracy:    {acc:.4f}  ({(pred[keep] == gt[keep]).sum()}/{n})")
    print(f"prediction dist:   {pred_dist}")
    print(f"ground-truth dist: {gt_dist}")

    # ---- Per-item details ----
    if args.per_item_output:
        with open(args.per_item_output, "w") as f:
            for i, r in enumerate(samples):
                if not valid[i]:
                    continue
                row_sims = sims[i].tolist()  # length-4
                # Sort options by similarity, descending
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
                    "predicted": str(pred[i]),
                    "correct": bool(pred[i] == correct_letters[i]),
                    "metadata": r.get("metadata"),
                    "ranking": ranking,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote per-item details to %s", args.per_item_output)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model_id": args.model_id,
                "split": args.split,
                "n_items": int(n),
                "accuracy": float(acc),
                "pred_dist": pred_dist,
                "gt_dist": gt_dist,
            }, f, indent=2)
        logger.info("Wrote summary metrics to %s", args.output)


if __name__ == "__main__":
    main()
