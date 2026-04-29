# ruff: noqa: ANN001, ANN201, E741
"""Smoke-test script for the 32 kHz BEATs checkpoint.

Loads BEATs_iter3_plus_32khz_e20 via the standard avex.load_model() path,
extracts embeddings on xeno-canto files at their native 32 kHz sample rate,
and runs clustering + retrieval metrics as a quality sanity check.

Also loads the standard 16 kHz BEATs_iter3_plus_AS2M for comparison.
"""

import glob
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_DIR = "/mnt/home/esp-data/test_copy_32k"
MAX_DURATION_S = 5.0
N_FILES = 30


def _label_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    stem = re.sub(r"^XC\d+-", "", stem)
    for prefix in [
        "Phaneroptera_falcata",
        "P_falcata",
        "Barbitistes_ocskayi",
        "Barbitistes",
        "Poecilimon_zimmeri",
        "Poecilimon_thessalicus",
        "Isophya",
        "Pholidoptera",
        "Gryllus",
        "Cicada",
        "Insect",
        "Northern",
    ]:
        if stem.startswith(prefix):
            return "Phaneroptera_falcata" if prefix == "P_falcata" else prefix.split("_")[0]
    return "other"


def load_audio_files(target_sr: int):
    wav_paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))[:N_FILES]
    waveforms, labels = [], []
    max_samples = int(MAX_DURATION_S * target_sr)
    for p in wav_paths:
        try:
            wav, sr = torchaudio.load(p)
        except Exception:
            continue
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav[:, :max_samples]
        if wav.shape[1] < target_sr:
            continue
        waveforms.append(wav.squeeze(0))
        labels.append(_label_from_filename(os.path.basename(p)))
    logger.info(f"Loaded {len(waveforms)} files at {target_sr} Hz")
    return waveforms, labels


@torch.no_grad()
def extract_embeddings(model, waveforms):
    embeddings = []
    for wav in waveforms:
        x = wav.unsqueeze(0)
        feats = model(x)
        if feats.dim() == 3:
            emb = feats.mean(dim=1)
        else:
            emb = feats
        embeddings.append(emb.squeeze(0).cpu())
    return torch.stack(embeddings)


def compute_metrics(embeddings: torch.Tensor, labels: list[str]) -> dict:
    from avex.evaluation.clustering import eval_clustering
    from avex.evaluation.retrieval import eval_retrieval

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    int_labels = torch.tensor([label_to_idx[l] for l in labels])
    counts = torch.bincount(int_labels)
    valid_classes = set((counts >= 2).nonzero(as_tuple=True)[0].tolist())
    mask_t = torch.tensor([label_to_idx[l] in valid_classes for l in labels])
    emb_f = embeddings[mask_t]
    lab_f = int_labels[mask_t]
    n_clusters = len(set(lab_f.tolist()))

    metrics = {}
    try:
        metrics.update(eval_clustering(emb_f, lab_f, n_clusters=n_clusters))
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
    try:
        metrics.update(eval_retrieval(emb_f, lab_f, batch_size=512))
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
    return metrics


def main():
    import avex

    results = {}

    # ── 32 kHz model ──────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("BEATs 32 kHz (BEATs_iter3_plus_32khz_e20)")
    logger.info("=" * 72)

    waveforms_32k, labels = load_audio_files(target_sr=32000)
    if len(waveforms_32k) < 10:
        logger.error("Not enough audio files")
        sys.exit(1)

    t0 = time.time()
    model_32k = avex.load_model(
        "BEATs_iter3_plus_32khz_e20",
        device="cpu",
        return_features_only=True,
    )
    model_32k.eval()
    logger.info(f"  Loaded in {time.time() - t0:.1f}s")

    cfg = model_32k.backbone.cfg
    logger.info(
        f"  Config: sample_frequency={cfg.sample_frequency}, "
        f"num_mel_bins={cfg.num_mel_bins}, "
        f"fbank_mean={cfg.fbank_mean:.5f}, fbank_std={cfg.fbank_std:.5f}, "
        f"deep_norm={cfg.deep_norm}"
    )

    t0 = time.time()
    emb_32k = extract_embeddings(model_32k, waveforms_32k)
    logger.info(f"  Embeddings extracted in {time.time() - t0:.1f}s  shape={emb_32k.shape}")

    norms = emb_32k.norm(dim=1)
    logger.info(f"  Norm: {norms.mean():.2f} +/- {norms.std():.2f}")
    assert not torch.isnan(emb_32k).any(), "32 kHz embeddings contain NaN!"
    assert not torch.isinf(emb_32k).any(), "32 kHz embeddings contain Inf!"
    assert norms.mean() > 0.1, "32 kHz embedding norms suspiciously small"

    metrics_32k = compute_metrics(emb_32k, labels)
    for k, v in sorted(metrics_32k.items()):
        logger.info(f"  {k}: {v:.4f}")
    results["BEATs_32kHz"] = metrics_32k

    del model_32k

    # ── 16 kHz baseline ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("BEATs 16 kHz baseline (BEATs_iter3_plus_AS2M)")
    logger.info("=" * 72)

    waveforms_16k, _ = load_audio_files(target_sr=16000)

    t0 = time.time()
    model_16k = avex.load_model(
        "BEATs_iter3_plus_AS2M",
        device="cpu",
        return_features_only=True,
    )
    model_16k.eval()
    logger.info(f"  Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    emb_16k = extract_embeddings(model_16k, waveforms_16k)
    logger.info(f"  Embeddings extracted in {time.time() - t0:.1f}s  shape={emb_16k.shape}")

    norms = emb_16k.norm(dim=1)
    logger.info(f"  Norm: {norms.mean():.2f} +/- {norms.std():.2f}")

    metrics_16k = compute_metrics(emb_16k, labels)
    for k, v in sorted(metrics_16k.items()):
        logger.info(f"  {k}: {v:.4f}")
    results["BEATs_16kHz"] = metrics_16k

    del model_16k

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    all_keys = sorted(set().union(*(m.keys() for m in results.values())))
    header = f"{'Model':<25}" + "".join(f"{k:>20}" for k in all_keys)
    logger.info(header)
    logger.info("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<25}" + "".join(f"{metrics.get(k, float('nan')):>20.4f}" for k in all_keys)
        logger.info(row)


if __name__ == "__main__":
    main()
