# ruff: noqa: ANN001, ANN201, DOC201, E741, F821
"""Comprehensive test script for all official models after BEATs loading fixes.

1. EfficientNet / EAT models: verify changes don't affect them (no regression).
2. BEATs models: verify they load correctly and produce sensible embeddings.

Run on CPU with a small subset of xeno-canto audio files.
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

EFFNET_MODELS = [
    "esp_aves2_effnetb0_all",
    "esp_aves2_effnetb0_bio",
    "esp_aves2_effnetb0_audioset",
]

EAT_MODELS = [
    "esp_aves2_eat_all",
    "esp_aves2_eat_bio",
    "esp_aves2_sl_eat_all_ssl_all",
    "esp_aves2_sl_eat_bio_ssl_all",
]

BEATS_MODELS = [
    "esp_aves2_sl_beats_all",
    "esp_aves2_sl_beats_bio",
    "esp_aves2_naturelm_audio_v1_beats",
    "BEATs_iter3_plus_AS2M",
    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2",
]


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
            if prefix == "P_falcata":
                return "Phaneroptera_falcata"
            return prefix.split("_")[0]
    return "other"


def load_audio_files(audio_dir: str, max_files: int, target_sr: int, max_dur_s: float):
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
        if wav.shape[1] < target_sr:
            continue
        waveforms.append(wav.squeeze(0))
        labels.append(_label_from_filename(os.path.basename(p)))
        filenames.append(os.path.basename(p))

    return waveforms, labels, filenames


@torch.no_grad()
def extract_embeddings(model, waveforms, device="cpu"):
    embeddings = []
    for wav in waveforms:
        x = wav.unsqueeze(0).to(device)
        feats = model(x)
        if feats.dim() == 4:
            emb = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:
            emb = feats.mean(dim=1)
        else:
            emb = feats
        embeddings.append(emb.squeeze(0).cpu())
    return torch.stack(embeddings)


def embedding_sanity_check(name: str, embeddings: torch.Tensor) -> dict:
    """Return diagnostic dict; flag problems."""
    norms = embeddings.norm(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    off_diag = cos_sim[mask]

    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()
    all_same = (embeddings.std(dim=0).mean() < 1e-6).item()

    return {
        "name": name,
        "shape": tuple(embeddings.shape),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "cos_mean": off_diag.mean().item(),
        "cos_std": off_diag.std().item(),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "all_same": all_same,
        "healthy": not has_nan and not has_inf and not all_same and norms.mean() > 0.1,
    }


def compute_metrics(embeddings: torch.Tensor, labels: list[str]) -> dict:
    from avex.evaluation.clustering import eval_clustering
    from avex.evaluation.retrieval import eval_retrieval

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    int_labels = torch.tensor([label_to_idx[l] for l in labels])

    counts = torch.bincount(int_labels)
    valid_classes = set((counts >= 2).nonzero(as_tuple=True)[0].tolist())
    mask_t = torch.tensor([label_to_idx[l] in valid_classes for l in labels])
    emb_filtered = embeddings[mask_t]
    labels_filtered = int_labels[mask_t]
    n_clusters = len(set(labels_filtered.tolist()))

    metrics = {}
    try:
        metrics.update(eval_clustering(emb_filtered, labels_filtered, n_clusters=n_clusters))
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
    try:
        metrics.update(eval_retrieval(emb_filtered, labels_filtered, batch_size=512))
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
    return metrics


def test_model_group(model_names, waveforms, labels, n_files, group_label, do_metrics=False):
    """Load models, extract embeddings, run checks. Returns list of result dicts."""
    import avex

    results = []
    for model_name in model_names:
        logger.info("")
        logger.info(f"--- {model_name} ---")
        t0 = time.time()
        try:
            model = avex.load_model(model_name, device="cpu", return_features_only=True)
            model.eval()
            load_time = time.time() - t0
            logger.info(f"Loaded in {load_time:.1f}s")
        except Exception as e:
            logger.error(f"FAILED to load: {e}")
            results.append({"name": model_name, "error": str(e)})
            continue

        t0 = time.time()
        try:
            embs = extract_embeddings(model, waveforms[:n_files], device="cpu")
            infer_time = time.time() - t0
            logger.info(f"Embeddings extracted in {infer_time:.1f}s ({n_files} files)")
        except Exception as e:
            logger.error(f"FAILED forward pass: {e}")
            results.append({"name": model_name, "error": str(e)})
            del model
            continue

        diag = embedding_sanity_check(model_name, embs)
        logger.info(
            f"  shape={diag['shape']}  norm={diag['norm_mean']:.2f}±{diag['norm_std']:.2f}  "
            f"cos={diag['cos_mean']:.3f}±{diag['cos_std']:.3f}  "
            f"nan={diag['has_nan']}  inf={diag['has_inf']}  "
            f"healthy={diag['healthy']}"
        )

        if do_metrics and diag["healthy"]:
            metrics = compute_metrics(embs, labels[:n_files])
            diag["metrics"] = metrics
            for k, v in sorted(metrics.items()):
                logger.info(f"  {k}: {v:.4f}")

        results.append(diag)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    logger.info("Loading audio files ...")
    waveforms, labels, filenames = load_audio_files(AUDIO_DIR, 40, TARGET_SR, MAX_DURATION_S)
    logger.info(f"Loaded {len(waveforms)} files")
    if len(waveforms) < 5:
        logger.error("Not enough audio files. Aborting.")
        sys.exit(1)

    all_results = {}

    # ---- EfficientNet (regression check, 5 files each) ----
    logger.info("")
    logger.info("=" * 72)
    logger.info("EFFICIENTNET MODELS (regression check — should be unaffected)")
    logger.info("=" * 72)
    all_results["efficientnet"] = test_model_group(
        EFFNET_MODELS, waveforms, labels, n_files=5, group_label="EfficientNet"
    )

    # ---- EAT (regression check, 5 files each) ----
    logger.info("")
    logger.info("=" * 72)
    logger.info("EAT MODELS (regression check — should be unaffected)")
    logger.info("=" * 72)
    all_results["eat"] = test_model_group(EAT_MODELS, waveforms, labels, n_files=5, group_label="EAT")

    # ---- BEATs (correctness check, 30 files with metrics) ----
    logger.info("")
    logger.info("=" * 72)
    logger.info("BEATS MODELS (correctness check — should produce good embeddings)")
    logger.info("=" * 72)
    all_results["beats"] = test_model_group(
        BEATS_MODELS, waveforms, labels, n_files=30, group_label="BEATs", do_metrics=True
    )

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)

    for group, results in all_results.items():
        logger.info(f"\n  {group.upper()}:")
        for r in results:
            if "error" in r:
                logger.info(f"    {r['name']}: ERROR — {r['error']}")
            else:
                status = "OK" if r["healthy"] else "PROBLEM"
                extra = ""
                if "metrics" in r:
                    m = r["metrics"]
                    ari = m.get("clustering_ari", float("nan"))
                    p1 = m.get("retrieval_precision_at_1", float("nan"))
                    extra = f"  ARI={ari:.3f}  P@1={p1:.3f}"
                logger.info(f"    {r['name']}: {status}  norm={r['norm_mean']:.2f}  cos={r['cos_mean']:.3f}{extra}")

    # Final pass/fail
    all_ok = True
    for results in all_results.values():
        for r in results:
            if "error" in r or not r.get("healthy", False):
                all_ok = False
    logger.info(f"\nOverall: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
