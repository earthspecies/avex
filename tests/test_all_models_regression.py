# ruff: noqa: ANN001, ANN201, ANN202, DOC201, E741, F821
"""Regression + quality test for all official models after BEATs loading fixes.

For EfficientNet / EAT:
  - Loads the model with the FIXED code path
  - Loads the model with a manually-reproduced OLD code path (no new fixes)
  - Asserts embeddings are bitwise identical (proves zero regression)
  - Runs clustering / retrieval metrics (quality check)

For BEATs:
  - Loads with fixed code, runs quality metrics (already proven in prior test)
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
N_FILES = 30

NON_BEATS_MODELS = [
    "esp_aves2_effnetb0_all",
    "esp_aves2_effnetb0_bio",
    "esp_aves2_effnetb0_audioset",
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
            return "Phaneroptera_falcata" if prefix == "P_falcata" else prefix.split("_")[0]
    return "other"


def load_audio_files():
    wav_paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))[:N_FILES]
    waveforms, labels = [], []
    max_samples = int(MAX_DURATION_S * TARGET_SR)
    for p in wav_paths:
        try:
            wav, sr = torchaudio.load(p)
        except Exception:
            continue
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav[:, :max_samples]
        if wav.shape[1] < TARGET_SR:
            continue
        waveforms.append(wav.squeeze(0))
        labels.append(_label_from_filename(os.path.basename(p)))
    logger.info(f"Loaded {len(waveforms)} files")
    return waveforms, labels


@torch.no_grad()
def extract_embeddings(model, waveforms):
    embeddings = []
    for wav in waveforms:
        x = wav.unsqueeze(0)
        feats = model(x)
        if feats.dim() == 4:
            # EfficientNet spatial features (B, C, H, W) → global avg pool → (B, C)
            emb = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:
            # BEATs/EAT temporal features (B, T, D) → mean pool → (B, D)
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


# ---------------------------------------------------------------------------
# Old-path loading: reproduces the ORIGINAL _load_checkpoint (pre-fix)
# to prove non-BEATs models get identical results.
# ---------------------------------------------------------------------------


def _old_process_state_dict(state_dict, keep_classifier=False, drop_model_prefix=True):
    """Original _process_state_dict WITHOUT the 'elif model' fix."""
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # NOTE: no elif "model" check -- this is the old behavior

    if not keep_classifier:
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        state_dict.pop("model.classifier.1.weight", None)
        state_dict.pop("model.classifier.1.bias", None)

    processed = {}
    for key, value in state_dict.items():
        pk = key
        if pk.startswith("module."):
            pk = pk[7:]
        elif drop_model_prefix and pk.startswith("model."):
            pk = pk[6:]
        if not keep_classifier and any(t in pk.lower() for t in ["classifier", "head", "classification"]):
            continue
        processed[pk] = value
    return processed


def _old_load_checkpoint(model, checkpoint_path, device, keep_classifier=False):
    """Original _load_checkpoint WITHOUT the backbone-prefix fix."""
    from avex.io import anypath
    from avex.utils.utils import universal_torch_load

    checkpoint = universal_torch_load(anypath(checkpoint_path), map_location=device)
    target_keys = model.state_dict().keys()
    target_has_model_prefix = any(k.startswith("model.") for k in target_keys)
    state_dict = _old_process_state_dict(
        checkpoint,
        keep_classifier=keep_classifier,
        drop_model_prefix=not target_has_model_prefix,
    )
    # NOTE: no backbone. prefix adaptation -- this is the old behavior
    model.load_state_dict(state_dict, strict=False)


def load_model_old_path(model_name: str):
    """Load a non-BEATs model using the OLD code path (no fixes)."""
    from avex.models.utils.factory import build_model_from_spec
    from avex.models.utils.registry import get_checkpoint_path, get_model_spec

    spec = get_model_spec(model_name)
    ckpt = get_checkpoint_path(model_name)

    # Reproduce the old _load_from_modelspec behavior
    spec.device = "cpu"
    if ckpt:
        spec.pretrained = False

    model = build_model_from_spec(spec, "cpu", return_features_only=True)
    if ckpt:
        _old_load_checkpoint(model, ckpt, "cpu", keep_classifier=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    waveforms, labels = load_audio_files()
    if len(waveforms) < 10:
        logger.error("Not enough audio files")
        sys.exit(1)

    import avex

    results = {}

    # ==== NON-BEATS: regression proof + quality ====
    logger.info("=" * 72)
    logger.info("NON-BEATS MODELS: regression check (old vs new must be identical)")
    logger.info("=" * 72)

    for name in NON_BEATS_MODELS:
        logger.info("")
        logger.info(f"--- {name} ---")

        # Load with FIXED code
        t0 = time.time()
        try:
            model_new = avex.load_model(name, device="cpu", return_features_only=True)
            model_new.eval()
        except Exception as e:
            logger.error(f"  FIXED load failed: {e}")
            results[name] = {"error": str(e)}
            continue

        # Load with OLD code
        try:
            model_old = load_model_old_path(name)
        except Exception as e:
            logger.error(f"  OLD load failed: {e}")
            results[name] = {"error": str(e)}
            del model_new
            continue

        load_time = time.time() - t0
        logger.info(f"  Both models loaded in {load_time:.1f}s")

        # Compare weights directly
        params_identical = True
        for (k1, v1), (k2, v2) in zip(
            sorted(model_new.state_dict().items()),
            sorted(model_old.state_dict().items()),
            strict=False,
        ):
            if k1 != k2 or not torch.equal(v1, v2):
                params_identical = False
                logger.warning(f"  Parameter mismatch: {k1} vs {k2}")
                break

        # Extract embeddings from both
        t0 = time.time()
        emb_new = extract_embeddings(model_new, waveforms)
        emb_old = extract_embeddings(model_old, waveforms)
        infer_time = time.time() - t0
        logger.info(f"  Embeddings extracted in {infer_time:.1f}s ({len(waveforms)} files x2)")

        embeddings_identical = torch.equal(emb_new, emb_old)
        max_diff = (emb_new - emb_old).abs().max().item()

        logger.info(f"  Parameters identical: {params_identical}")
        logger.info(f"  Embeddings identical: {embeddings_identical} (max diff: {max_diff:.2e})")

        if not embeddings_identical:
            logger.warning(f"  *** REGRESSION: embeddings differ by up to {max_diff:.2e} ***")

        # Quality metrics (on the fixed model — identical anyway)
        metrics = compute_metrics(emb_new, labels)
        norms = emb_new.norm(dim=1)

        logger.info(f"  Norm: {norms.mean():.2f} ± {norms.std():.2f}")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        results[name] = {
            "params_identical": params_identical,
            "embeddings_identical": embeddings_identical,
            "max_diff": max_diff,
            "metrics": metrics,
            "norm_mean": norms.mean().item(),
        }

        del model_new, model_old

    # ==== BEATS: quality check ====
    logger.info("")
    logger.info("=" * 72)
    logger.info("BEATS MODELS: quality check")
    logger.info("=" * 72)

    for name in BEATS_MODELS:
        logger.info("")
        logger.info(f"--- {name} ---")
        t0 = time.time()
        try:
            model = avex.load_model(name, device="cpu", return_features_only=True)
            model.eval()
            logger.info(f"  Loaded in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"  Load failed: {e}")
            results[name] = {"error": str(e)}
            continue

        t0 = time.time()
        emb = extract_embeddings(model, waveforms)
        logger.info(f"  Embeddings in {time.time() - t0:.1f}s")

        metrics = compute_metrics(emb, labels)
        norms = emb.norm(dim=1)
        logger.info(f"  Norm: {norms.mean():.2f} ± {norms.std():.2f}")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        results[name] = {"metrics": metrics, "norm_mean": norms.mean().item()}
        del model

    # ==== SUMMARY ====
    logger.info("")
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    header = f"{'Model':<50} {'Identical':>9} {'ARI':>7} {'P@1':>7} {'ROC':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    all_ok = True
    for name in NON_BEATS_MODELS + BEATS_MODELS:
        r = results.get(name, {})
        if "error" in r:
            logger.info(f"{name:<50} {'ERROR':>9}")
            all_ok = False
            continue

        identical = r.get("embeddings_identical", "n/a")
        m = r.get("metrics", {})
        ari = m.get("clustering_ari", float("nan"))
        p1 = m.get("retrieval_precision_at_1", float("nan"))
        roc = m.get("retrieval_roc_auc", float("nan"))

        id_str = str(identical) if isinstance(identical, bool) else identical
        logger.info(f"{name:<50} {id_str:>9} {ari:>7.3f} {p1:>7.3f} {roc:>7.3f}")

        if isinstance(identical, bool) and not identical:
            all_ok = False

    logger.info("")
    logger.info(f"Overall: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
