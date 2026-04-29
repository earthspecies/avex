"""Miniature CBI test for a trained CLAP checkpoint.

What this measures: nearest-neighbour audio-embedding classification on a small
slice of the BEANS `cbi_test` split. We pick the N most-frequent classes,
take K samples per class, encode all audios with the CLAP audio encoder, then
for each sample find its nearest neighbour (cosine similarity, excluding itself)
and report whether the prediction matches the true class.

Why this and not zero-shot text: the CBI labels are 6-letter eBird banding codes
(e.g. ``aldfly``); CLAP has never seen those tokens. A small audio-embedding
1-NN sweep is the cheapest sanity check that the trained audio encoder
actually clusters CBI species.
"""

import argparse
import logging
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("cbi_mini")
logger.setLevel(logging.INFO)


def load_clap(checkpoint_path: str, run_dir_config: str, device: torch.device):
    """Build a CLIP model from the run's saved config, load the checkpoint state."""
    from avex.configs import RunConfig
    from avex.models.get_model import get_model

    with open(run_dir_config) as f:
        raw = yaml.safe_load(f)
    # Saved configs may have device='cuda:0'; normalize before validation
    if "model_spec" in raw and isinstance(raw["model_spec"], dict):
        raw["model_spec"]["device"] = "cuda" if device.type == "cuda" else "cpu"
    cfg = RunConfig(**raw)
    cfg.model_spec.device = "cuda" if device.type == "cuda" else "cpu"
    # Skip transferring ESP weights at build time — we'll overwrite from checkpoint.
    cfg.model_spec.audio_encoder_init_from = None

    model = get_model(cfg.model_spec, num_classes=0).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded checkpoint epoch=%s, missing=%d, unexpected=%d",
        ckpt.get("epoch"),
        len(missing),
        len(unexpected),
    )
    model.eval()
    return model, cfg


def build_subset(
    n_classes: int,
    samples_per_class: int,
) -> tuple[list[dict], list[str]]:
    """Sample a balanced subset of cbi_test."""
    from esp_data.io import anypath
    from esp_data.io.filesystem import filesystem_from_path
    import json

    cbi_test_path = "gs://esp-ml-datasets/beans/v0.1.0/raw/cbi_test.jsonl"
    fs = filesystem_from_path(cbi_test_path)
    rows = []
    with fs.open(cbi_test_path, "rb") as f:
        for line in f:
            rows.append(json.loads(line))

    # Pick N most-frequent classes
    label_counts = Counter(r["label"] for r in rows)
    top = [lbl for lbl, _ in label_counts.most_common(n_classes)]

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] in top:
            by_label[r["label"]].append(r)

    subset: list[dict] = []
    for lbl in top:
        subset.extend(by_label[lbl][:samples_per_class])
    return subset, top


def load_audio(local_path: str, target_sr: int, target_seconds: int) -> np.ndarray:
    """Load + window/pad audio to (target_sr * target_seconds) samples."""
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio
    import librosa

    full_path = anypath("gs://esp-ml-datasets/beans/v0.1.0/raw/") / local_path
    audio, sr = read_audio(full_path)
    audio = audio.astype(np.float32)
    audio = audio_stereo_to_mono(audio, mono_method="average")
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
    target_len = target_sr * target_seconds
    if audio.shape[0] >= target_len:
        # center crop
        start = (audio.shape[0] - target_len) // 2
        audio = audio[start : start + target_len]
    else:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    return audio


def encode_audios(model, samples: list[dict], target_sr: int, target_seconds: int, device: torch.device,
                  batch_size: int = 8) -> torch.Tensor:
    """Encode all subset samples to normalized audio embeddings."""
    embs = []
    n = len(samples)
    for start in range(0, n, batch_size):
        batch = samples[start : start + batch_size]
        wavs = []
        for r in batch:
            wavs.append(load_audio(r["local_path"], target_sr, target_seconds))
        wav_t = torch.from_numpy(np.stack(wavs)).to(device)
        pad_mask = torch.zeros_like(wav_t, dtype=torch.bool)
        with torch.no_grad():
            emb = model.encode_audio(wav_t, pad_mask)  # already F.normalized
        embs.append(emb.cpu())
        logger.info("Encoded %d / %d", min(start + batch_size, n), n)
    return torch.cat(embs, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="/mnt/home/avex/runs/clap/clap_chain_synthetic_xc_inat/"
                           "2026-04-28_06-55-37/best_model.pt")
    p.add_argument("--config",
                   default="/mnt/home/avex/runs/clap/clap_chain_synthetic_xc_inat/"
                           "2026-04-28_06-55-37/config.yml")
    p.add_argument("--n-classes", type=int, default=20)
    p.add_argument("--samples-per-class", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Building model + loading checkpoint...")
    t0 = time.time()
    model, cfg = load_clap(args.checkpoint, args.config, device)
    logger.info("Model ready in %.1fs (%d params)", time.time() - t0,
                sum(p.numel() for p in model.parameters()))

    target_sr = cfg.model_spec.audio_config.sample_rate
    target_seconds = cfg.model_spec.audio_config.target_length_seconds

    logger.info("Sampling balanced subset: %d classes x %d samples = %d clips",
                args.n_classes, args.samples_per_class,
                args.n_classes * args.samples_per_class)
    samples, classes = build_subset(args.n_classes, args.samples_per_class)
    labels = [s["label"] for s in samples]
    logger.info("Got %d clips across %d classes", len(samples), len(set(labels)))

    logger.info("Encoding audios...")
    t0 = time.time()
    embs = encode_audios(model, samples, target_sr, target_seconds, device, args.batch_size)
    logger.info("Encoded %d clips in %.1fs (shape=%s)", len(samples), time.time()-t0, tuple(embs.shape))

    # 1-NN accuracy: cosine similarity (embs are already normalized), exclude self
    sims = embs @ embs.T
    sims.fill_diagonal_(-1.0)
    top5 = sims.topk(5, dim=1).indices  # (N, 5)
    label_arr = np.array(labels)

    top1_correct = 0
    top5_correct = 0
    chance = 1.0 / args.n_classes
    n = len(samples)
    for i in range(n):
        nn_labels = label_arr[top5[i].numpy()]
        if nn_labels[0] == label_arr[i]:
            top1_correct += 1
        if label_arr[i] in nn_labels:
            top5_correct += 1

    print()
    print("=" * 60)
    print("CBI MINI TEST RESULTS")
    print("=" * 60)
    print(f"checkpoint:           {args.checkpoint}")
    print(f"n_classes:            {args.n_classes}")
    print(f"samples_per_class:    {args.samples_per_class}")
    print(f"total samples:        {n}")
    print(f"chance (1/n_classes): {chance:.3f}")
    print()
    print(f"1-NN top-1 accuracy:  {top1_correct/n:.3f}  ({top1_correct}/{n})")
    print(f"1-NN top-5 accuracy:  {top5_correct/n:.3f}  ({top5_correct}/{n})")
    print(f"top-1 lift over chance: {(top1_correct/n) / chance:.1f}x")


if __name__ == "__main__":
    main()
