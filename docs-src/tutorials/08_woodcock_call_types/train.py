"""Train linear probes for Eurasian woodcock call types.

This script mirrors the woodcock notebooks as a reproducible pipeline:

- Download Zenodo assets (`selections.csv`, `selections_wavs.zip`)
- Extract WAVs (3-second selections)
- Build `data/metadata.csv`
- Extract embeddings for two avex models (BEATs + EfficientNet-B0)
- Evaluate linear probes with:
  - stratified random split (baseline)
  - leave-one-deployment-out (LODO) cross-validation (site generalisation)

Usage
-----
```bash
uv run python examples/08_woodcock_call_types/train.py --config examples/08_woodcock_call_types/config.yaml
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from avex import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from tqdm.auto import tqdm

from utils.probing import compute_training_free_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train woodcock call-type probes and save embeddings/metrics.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def find_repo_root(start: pathlib.Path) -> pathlib.Path:
    """Find the repository root by searching for `pyproject.toml`.

    Returns
    -------
    pathlib.Path
        Path to the repository root directory.

    Raises
    ------
    FileNotFoundError
        If no `pyproject.toml` is found in *start* or any of its parents.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    msg = "Could not locate repo root (pyproject.toml not found)."
    raise FileNotFoundError(msg)


def load_config(path: pathlib.Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    ValueError
        If the YAML file does not contain a top-level mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        msg = f"Invalid config structure in {path}"
        raise ValueError(msg)
    return cfg


def download_with_progress(url: str, dest: pathlib.Path) -> None:
    """Download *url* to *dest*, printing a simple progress indicator."""

    def _hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  {pct:5.1f}%", end="", flush=True)
        else:
            print(f"\r  {count * block_size / 1e6:.1f} MB", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def ensure_zenodo_assets(cfg: dict[str, Any], example_dir: pathlib.Path) -> None:
    """Download selections CSV + WAV zip if missing."""
    record = str(cfg["dataset"]["zenodo_record"])
    base = f"https://zenodo.org/records/{record}/files"

    selections_csv = example_dir / str(cfg["dataset"]["selections_csv"])
    audio_zip = example_dir / str(cfg["dataset"]["audio_zip"])

    min_bytes: dict[str, int] = {
        # CSV should never be empty if the download succeeded.
        "selections.csv": 1024,
        # ZIP is ~559MB; treat anything tiny as partial/corrupt.
        "selections_wavs.zip": 100 * 1024 * 1024,
    }

    for fname, path in [
        ("selections.csv", selections_csv),
        ("selections_wavs.zip", audio_zip),
    ]:
        if path.exists() and path.stat().st_size >= min_bytes[fname]:
            continue
        if path.exists():
            path.unlink()
        print(f"Downloading {fname} ...")
        download_with_progress(f"{base}/{fname}", path)


def ensure_audio_extracted(cfg: dict[str, Any], example_dir: pathlib.Path) -> None:
    """Extract the audio zip into the audio directory."""
    audio_zip = example_dir / str(cfg["dataset"]["audio_zip"])
    audio_dir = example_dir / str(cfg["dataset"]["audio_dir"])
    sentinel = audio_dir / ".extracted"
    audio_dir.mkdir(parents=True, exist_ok=True)
    if sentinel.exists():
        return
    print(f"Extracting {audio_zip} ...")
    with zipfile.ZipFile(audio_zip, "r") as zf:
        zf.extractall(audio_dir)
    sentinel.touch()


def simplify_call_type(annotation: str) -> str:
    """Map raw Raven Pro annotation to one of three primary call types.

    Returns
    -------
    str
        One of ``"whistle"``, ``"chase"``, or ``"croak"``.
    """
    ann = str(annotation).lower()
    if "whistle" in ann:
        return "whistle"
    if "chase" in ann:
        return "chase"
    return "croak"


def build_metadata(cfg: dict[str, Any], example_dir: pathlib.Path) -> pd.DataFrame:
    """Build `metadata.csv` from `selections.csv` and extracted WAVs.

    Returns
    -------
    pd.DataFrame
        One row per selection with columns including ``path``, ``call_type``,
        and ``deploy_id``.  Rows without a matching WAV on disk are excluded.
    """
    selections_csv = example_dir / str(cfg["dataset"]["selections_csv"])
    audio_dir = example_dir / str(cfg["dataset"]["audio_dir"])

    wav_files = sorted(audio_dir.rglob("*.wav"))
    wav_by_selec: dict[int, pathlib.Path] = {}
    for p in wav_files:
        # Zenodo asset `selections_wavs.zip` extracts files like `1234.wav`,
        # where the filename corresponds to the `selec` id in `selections.csv`.
        try:
            selec_id = int(p.stem)
        except ValueError:
            continue
        wav_by_selec[selec_id] = p

    df_raw = pd.read_csv(selections_csv)

    records: list[dict[str, Any]] = []
    for _, row in df_raw.iterrows():
        rec_base = pathlib.Path(str(row["sound.files"])).stem
        selec_id = int(row["selec"])
        wav_path = wav_by_selec.get(selec_id)
        records.append(
            {
                "selec": selec_id,
                "recording": rec_base,
                "deploy_id": row["deploy.id"],
                "start_s": float(row["start"]),
                "end_s": float(row["end"]),
                "duration_s": float(row["end"] - row["start"]),
                "bottom_freq_khz": float(row["bottom.freq"]),
                "top_freq_khz": float(row["top.freq"]),
                "annotation_raw": str(row["annotation"]),
                "call_type": simplify_call_type(str(row["annotation"])),
                "path": str(wav_path) if wav_path is not None else "",
            }
        )

    df = pd.DataFrame.from_records(records)
    # Keep only selections with audio on disk
    df = df[df["path"].astype(str) != ""].reset_index(drop=True)
    return df


def load_audio_tensor(path: str, target_sr: int) -> torch.Tensor:
    """Load a WAV, convert to mono, resample, return (1, T) float tensor.

    Returns
    -------
    torch.Tensor
        Audio tensor of shape ``(1, T)`` at *target_sr*.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav_mono = wav.mean(axis=1)
    if sr != target_sr:
        import librosa

        wav_mono = librosa.resample(wav_mono, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(wav_mono).unsqueeze(0)


def extract_embeddings(
    paths: list[str],
    model_name: str,
    target_sr: int,
    device: str,
) -> np.ndarray:
    """Extract mean-pooled embeddings for each file path.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, embedding_dim)``.
    """
    model = load_model(model_name, return_features_only=True, device=device)
    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for p in tqdm(paths, desc=model_name):
            wav = load_audio_tensor(p, target_sr=target_sr)
            feats = model(wav)
            if feats.ndim == 3:
                # (1, T, D)
                pooled = feats.mean(dim=1).squeeze(0)
            else:
                # (1, C, H, W)
                pooled = feats.mean(dim=(2, 3)).squeeze(0)
            embs.append(pooled.cpu().numpy())
    del model
    return np.stack(embs)


def extract_beats_all_layers_avg(
    paths: list[str],
    model_name: str,
    target_sr: int,
    device: str,
) -> np.ndarray:
    """Extract BEATs embeddings by averaging mean-pooled outputs across all layers.

    Note
    ----
    This function is kept for backward compatibility with earlier notebooks/scripts.
    Newer code should prefer `extract_beats_all_layers` to persist per-layer outputs.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, embedding_dim)`` with one averaged embedding
        per audio clip.
    """
    all_layers = extract_beats_all_layers(
        paths=paths,
        model_name=model_name,
        target_sr=target_sr,
        device=device,
    )
    return all_layers.mean(axis=0)


def extract_beats_all_layers(
    paths: list[str],
    model_name: str,
    target_sr: int,
    device: str,
) -> np.ndarray:
    """Extract mean-pooled BEATs embeddings for every encoder layer.

    Parameters
    ----------
    paths : list[str]
        Audio file paths.
    model_name : str
        avex model name for SL-BEATs.
    target_sr : int
        Target sample rate.
    device : str
        Torch device string.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_layers, n_samples, embedding_dim)`` containing mean-pooled
        embeddings for each encoder layer.
    """
    model = load_model(model_name, return_features_only=True, device=device)
    model.eval()

    try:
        encoder_layers = model.model.encoder.layers
    except AttributeError:  # pragma: no cover
        encoder_layers = model.backbone.encoder.layers

    n_layers = len(encoder_layers)
    layer_store: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(idx: int) -> Callable[..., None]:
        def _hook(
            _module: torch.nn.Module,
            _inp: tuple[torch.Tensor, ...],
            out: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            layer_store[idx] = out[0] if isinstance(out, tuple) else out

        return _hook

    for i, layer in enumerate(encoder_layers):
        hooks.append(layer.register_forward_hook(_make_hook(i)))

    per_layer_embs: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for p in tqdm(paths, desc=f"{model_name} (all layers)"):
            layer_store.clear()
            wav = load_audio_tensor(p, target_sr=target_sr).to(device)
            _ = model(wav)
            for i in range(n_layers):
                out = layer_store[i]
                pooled = out.view(-1, out.shape[-1]).mean(dim=0)
                per_layer_embs[i].append(pooled.cpu().numpy())

    for h in hooks:
        h.remove()
    del model

    return np.stack([np.stack(v) for v in per_layer_embs])


def eval_random_split(
    X: np.ndarray, y: np.ndarray, test_size: float, random_state: int, max_iter: int
) -> dict[str, Any]:
    """Evaluate balanced accuracy on a fixed stratified split.

    Returns
    -------
    dict[str, Any]
        Dictionary with key ``balanced_accuracy``.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(sss.split(X, y))
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X[tr_idx], y[tr_idx])
    y_pred = clf.predict(X[te_idx])
    bal_acc = float(balanced_accuracy_score(y[te_idx], y_pred))
    return {"balanced_accuracy": bal_acc}


def eval_lodo(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    max_iter: int,
) -> dict[str, Any]:
    """Evaluate leave-one-group-out (deployment) balanced accuracy.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``n_folds``, ``mean_balanced_accuracy``, and
        ``std_balanced_accuracy``.
    """
    logo = LeaveOneGroupOut()
    fold_scores: list[float] = []
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        y_te = y[te_idx]
        if len(np.unique(y_te)) < 2:
            # Skip folds without class diversity (rare chase-only/absent folds)
            continue
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
        clf.fit(X[tr_idx], y[tr_idx])
        y_pred = clf.predict(X[te_idx])
        fold_scores.append(float(balanced_accuracy_score(y_te, y_pred)))
    return {
        "n_folds": int(len(fold_scores)),
        "mean_balanced_accuracy": float(np.mean(fold_scores)) if fold_scores else float("nan"),
        "std_balanced_accuracy": float(np.std(fold_scores)) if fold_scores else float("nan"),
    }


def main() -> None:
    """Run the full woodcock pipeline."""
    args = parse_args()
    repo_root = find_repo_root(pathlib.Path(__file__).resolve())
    example_dir = repo_root / "examples" / "08_woodcock_call_types"
    cfg = load_config(
        (repo_root / args.config).resolve()
        if not pathlib.Path(args.config).is_absolute()
        else pathlib.Path(args.config)
    )

    ensure_zenodo_assets(cfg, example_dir)
    ensure_audio_extracted(cfg, example_dir)

    df = build_metadata(cfg, example_dir)
    meta_path = example_dir / str(cfg["dataset"]["metadata_csv"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_path, index=False)
    print(f"Saved metadata: {meta_path} ({len(df)} rows)")

    emb_dir = example_dir / str(cfg["embeddings"]["dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)
    beats_path = emb_dir / str(cfg["embeddings"]["beats_file"])
    beats_all_avg_path = emb_dir / "beats_all_layers_embeddings.npy"
    beats_all_layers_dir = emb_dir / "beats_all_layers"
    effnet_path = emb_dir / str(cfg["embeddings"]["effnet_file"])

    target_sr = int(cfg["audio"]["target_sr"])
    device = "cpu"

    paths = df["path"].astype(str).tolist()
    if beats_path.exists():
        beats_embs = np.load(beats_path)
    else:
        beats_embs = extract_embeddings(
            paths=paths,
            model_name=str(cfg["models"]["beats"]["name"]),
            target_sr=target_sr,
            device=device,
        )
        np.save(beats_path, beats_embs)

    beats_all_layers_dir.mkdir(parents=True, exist_ok=True)
    any_layer_exists = any(beats_all_layers_dir.glob("layer_*.npy"))
    if beats_all_avg_path.exists() and any_layer_exists:
        beats_all_embs = np.load(beats_all_avg_path)
    else:
        beats_layers = extract_beats_all_layers(
            paths=paths,
            model_name=str(cfg["models"]["beats"]["name"]),
            target_sr=target_sr,
            device=device,
        )
        for i in range(beats_layers.shape[0]):
            np.save(beats_all_layers_dir / f"layer_{i:02d}.npy", beats_layers[i])
        beats_all_embs = beats_layers.mean(axis=0)
        np.save(beats_all_avg_path, beats_all_embs)

    if effnet_path.exists():
        effnet_embs = np.load(effnet_path)
    else:
        effnet_embs = extract_embeddings(
            paths=paths,
            model_name=str(cfg["models"]["effnet"]["name"]),
            target_sr=target_sr,
            device=device,
        )
        np.save(effnet_path, effnet_embs)

    y = df[str(cfg["probe"]["label_col"])].astype(str).to_numpy()
    groups = df[str(cfg["probe"]["lodo_group_col"])].astype(str).to_numpy()
    test_size = float(cfg["probe"]["test_size"])
    random_state = int(cfg["probe"]["random_state"])
    max_iter = int(cfg["probe"]["max_iter"])

    training_free = {
        f"{cfg['models']['beats']['name']} (last layer)": compute_training_free_metrics(
            beats_embs, y, random_state=random_state
        ),
        f"{cfg['models']['beats']['name']} (all layers avg)": compute_training_free_metrics(
            beats_all_embs, y, random_state=random_state
        ),
        str(cfg["models"]["effnet"]["name"]): compute_training_free_metrics(effnet_embs, y, random_state=random_state),
    }

    metrics = {
        "n_rows": int(len(df)),
        "class_counts": df["call_type"].value_counts().to_dict(),
        "training_free": training_free,
        "beats_last_layer": {
            "random_split": eval_random_split(
                beats_embs, y, test_size=test_size, random_state=random_state, max_iter=max_iter
            ),
            "lodo": eval_lodo(beats_embs, y, groups, random_state=random_state, max_iter=max_iter),
        },
        "beats_all_layers_avg": {
            "random_split": eval_random_split(
                beats_all_embs, y, test_size=test_size, random_state=random_state, max_iter=max_iter
            ),
            "lodo": eval_lodo(beats_all_embs, y, groups, random_state=random_state, max_iter=max_iter),
        },
        "effnet": {
            "random_split": eval_random_split(
                effnet_embs, y, test_size=test_size, random_state=random_state, max_iter=max_iter
            ),
            "lodo": eval_lodo(effnet_embs, y, groups, random_state=random_state, max_iter=max_iter),
        },
    }

    out_dir = example_dir / str(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / str(cfg["output"]["metrics_json"])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
