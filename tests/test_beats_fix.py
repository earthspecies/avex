# ruff: noqa: ANN001, ANN201, DOC201, E741
"""Test script to verify the BEATs model loading fixes.

Compares "broken" (BEATsConfig defaults with deep_norm=False) vs "fixed"
(correct config from reference checkpoint with deep_norm=True) when loading
safetensors weights trained with deep_norm=True.

Uses xeno-canto audio files from esp-data for realistic evaluation.
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
TARGET_SR = 16000
MAX_DURATION_S = 5.0
MAX_FILES = 40


# ---------------------------------------------------------------------------
# 1. Audio loading
# ---------------------------------------------------------------------------


def _label_from_filename(fname: str) -> str:
    """Extract a coarse species/group label from a xeno-canto filename."""
    stem = Path(fname).stem
    stem = re.sub(r"^XC\d+-", "", stem)  # strip XC ID

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
            # Merge P_falcata into Phaneroptera_falcata
            if prefix == "P_falcata":
                return "Phaneroptera_falcata"
            return prefix.split("_")[0]  # genus-level grouping
    return "other"


def load_audio_files(audio_dir: str, max_files: int, target_sr: int, max_dur_s: float):
    """Load WAV files, resample to target_sr, truncate to max_dur_s."""
    wav_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))[:max_files]
    waveforms, labels, filenames = [], [], []
    max_samples = int(max_dur_s * target_sr)

    for p in wav_paths:
        try:
            wav, sr = torchaudio.load(p)
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")
            continue
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav[:, :max_samples]
        if wav.shape[1] < target_sr:  # skip files < 1s
            continue
        waveforms.append(wav.squeeze(0))
        labels.append(_label_from_filename(os.path.basename(p)))
        filenames.append(os.path.basename(p))

    logger.info(f"Loaded {len(waveforms)} files from {audio_dir}")
    label_set = sorted(set(labels))
    logger.info(f"Label distribution: { {l: labels.count(l) for l in label_set} }")
    return waveforms, labels, filenames


# ---------------------------------------------------------------------------
# 2. Build broken vs fixed BEATs models
# ---------------------------------------------------------------------------


def build_broken_beats_model(device: str = "cpu"):
    """Reproduce the OLD buggy behaviour: BEATsConfig() defaults + safetensors weights."""
    from avex.models.beats.beats import BEATs, BEATsConfig
    from avex.models.beats_model import Model as BeatsModelClass
    from avex.models.utils.load import _load_checkpoint
    from avex.models.utils.registry import get_checkpoint_path

    # 1. Build with DEFAULT config (the bug: deep_norm=False)
    cfg = BEATsConfig()
    logger.info(f"[BROKEN] BEATsConfig defaults: deep_norm={cfg.deep_norm}")
    backbone = BEATs(cfg)
    backbone.to(device)

    # 2. Wrap in Model shell (mimicking factory construction with pretrained=False)
    model = BeatsModelClass.__new__(BeatsModelClass)
    torch.nn.Module.__init__(model)
    from avex.models.base_model import ModelBase

    ModelBase.__init__(model, device=device, audio_config=None)
    model.disable_layerdrop = False
    model.use_naturelm = False
    model.fine_tuned = True
    model.backbone = backbone
    model._return_features_only = True
    model.register_module("classifier", None)
    model.num_classes = None

    # 3. Load the safetensors weights (same checkpoint the fixed path would use)
    ckpt_path = get_checkpoint_path("esp_aves2_sl_beats_all")
    logger.info(f"[BROKEN] Loading weights from: {ckpt_path}")
    _load_checkpoint(model, ckpt_path, device=device, keep_classifier=False)

    model.eval()
    return model


def build_fixed_beats_model(device: str = "cpu"):
    """Load model via the FIXED avex.load_model() path."""
    import avex

    logger.info("[FIXED] Loading esp_aves2_sl_beats_all via avex.load_model ...")
    model = avex.load_model("esp_aves2_sl_beats_all", device=device, return_features_only=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(model, waveforms, device="cpu"):
    """Run forward pass and mean-pool frame embeddings → (N, D)."""
    embeddings = []
    for i, wav in enumerate(waveforms):
        x = wav.unsqueeze(0).to(device)  # (1, T)
        feats = model(x)  # (1, T', D) in features-only mode
        if feats.dim() == 3:
            emb = feats.mean(dim=1)  # (1, D)
        else:
            emb = feats
        embeddings.append(emb.squeeze(0).cpu())
        if (i + 1) % 10 == 0:
            logger.info(f"  ... processed {i + 1}/{len(waveforms)}")
    return torch.stack(embeddings)  # (N, D)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------


def compute_metrics(embeddings: torch.Tensor, labels: list[str]):
    """Compute clustering + retrieval metrics."""
    from avex.evaluation.clustering import eval_clustering
    from avex.evaluation.retrieval import eval_retrieval

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    int_labels = torch.tensor([label_to_idx[l] for l in labels])

    # Filter to labels with ≥2 samples (needed for retrieval)
    counts = torch.bincount(int_labels)
    valid_classes = set((counts >= 2).nonzero(as_tuple=True)[0].tolist())
    mask = torch.tensor([label_to_idx[l] in valid_classes for l in labels])
    emb_filtered = embeddings[mask]
    labels_filtered = int_labels[mask]
    n_clusters = len(set(labels_filtered.tolist()))

    logger.info(f"  Using {mask.sum().item()}/{len(labels)} samples ({n_clusters} classes with ≥2 samples)")

    metrics = {}

    # Clustering
    try:
        clust = eval_clustering(emb_filtered, labels_filtered, n_clusters=n_clusters)
        metrics.update(clust)
    except Exception as e:
        logger.warning(f"  Clustering failed: {e}")

    # Retrieval
    try:
        ret = eval_retrieval(emb_filtered, labels_filtered, batch_size=512)
        metrics.update(ret)
    except Exception as e:
        logger.warning(f"  Retrieval failed: {e}")

    return metrics


def embedding_diagnostics(name: str, embeddings: torch.Tensor):
    """Print basic sanity checks on embeddings."""
    norms = embeddings.norm(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # Mask out diagonal for off-diagonal stats
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    off_diag = cos_sim[mask]

    logger.info(f"  [{name}] Embedding shape: {embeddings.shape}")
    logger.info(
        f"  [{name}] Norm — mean: {norms.mean():.4f}, std: {norms.std():.4f}, "
        f"min: {norms.min():.4f}, max: {norms.max():.4f}"
    )
    logger.info(f"  [{name}] Cosine sim (off-diag) — mean: {off_diag.mean():.4f}, std: {off_diag.std():.4f}")
    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()
    if has_nan or has_inf:
        logger.warning(f"  [{name}] *** Contains NaN={has_nan}, Inf={has_inf} ***")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cpu"

    # Load audio
    waveforms, labels, filenames = load_audio_files(AUDIO_DIR, MAX_FILES, TARGET_SR, MAX_DURATION_S)
    if len(waveforms) < 5:
        logger.error("Not enough audio files loaded. Aborting.")
        sys.exit(1)

    # Build models
    logger.info("=" * 70)
    logger.info("Building BROKEN model (BEATsConfig defaults, deep_norm=False)")
    logger.info("=" * 70)
    t0 = time.time()
    broken_model = build_broken_beats_model(device)
    logger.info(f"Broken model built in {time.time() - t0:.1f}s")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Building FIXED model (config from reference checkpoint)")
    logger.info("=" * 70)
    t0 = time.time()
    fixed_model = build_fixed_beats_model(device)
    logger.info(f"Fixed model built in {time.time() - t0:.1f}s")

    # Check deep_norm on both
    logger.info("")
    logger.info("Config comparison:")
    logger.info(
        f"  Broken model backbone deep_norm = "
        f"{broken_model.backbone.encoder.deep_norm if hasattr(broken_model.backbone.encoder, 'deep_norm') else 'N/A'}"
    )
    logger.info(
        f"  Fixed  model backbone deep_norm = "
        f"{fixed_model.backbone.encoder.deep_norm if hasattr(fixed_model.backbone.encoder, 'deep_norm') else 'N/A'}"
    )

    # Extract embeddings
    logger.info("")
    logger.info("=" * 70)
    logger.info("Extracting embeddings (BROKEN)")
    logger.info("=" * 70)
    t0 = time.time()
    emb_broken = extract_embeddings(broken_model, waveforms, device)
    logger.info(f"Done in {time.time() - t0:.1f}s")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Extracting embeddings (FIXED)")
    logger.info("=" * 70)
    t0 = time.time()
    emb_fixed = extract_embeddings(fixed_model, waveforms, device)
    logger.info(f"Done in {time.time() - t0:.1f}s")

    # Diagnostics
    logger.info("")
    logger.info("=" * 70)
    logger.info("Embedding diagnostics")
    logger.info("=" * 70)
    embedding_diagnostics("BROKEN", emb_broken)
    embedding_diagnostics("FIXED", emb_fixed)

    # Metrics
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation metrics (BROKEN)")
    logger.info("=" * 70)
    metrics_broken = compute_metrics(emb_broken, labels)
    for k, v in sorted(metrics_broken.items()):
        logger.info(f"  {k}: {v:.4f}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation metrics (FIXED)")
    logger.info("=" * 70)
    metrics_fixed = compute_metrics(emb_fixed, labels)
    for k, v in sorted(metrics_fixed.items()):
        logger.info(f"  {k}: {v:.4f}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<30} {'BROKEN':>10} {'FIXED':>10} {'Delta':>10}")
    logger.info("-" * 62)
    all_keys = sorted(set(list(metrics_broken.keys()) + list(metrics_fixed.keys())))
    for k in all_keys:
        b = metrics_broken.get(k, float("nan"))
        f = metrics_fixed.get(k, float("nan"))
        logger.info(f"{k:<30} {b:>10.4f} {f:>10.4f} {f - b:>+10.4f}")


if __name__ == "__main__":
    main()
