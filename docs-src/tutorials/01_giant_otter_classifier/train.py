"""Train linear and attention probes on giant otter call type embeddings.

Downloads the giant otter vocalization dataset (21 call types) from the
Internet Archive, extracts embeddings using two avex models
(``esp_aves2_sl_beats_all`` and ``esp_aves2_effnetb0_all``), fits a
logistic-regression probe for each model, fits an avex ``AttentionProbe`` on
the two BEATs embedding sets, and saves accuracy results plus UMAP figures
(interactive HTML and static PNG).

Usage
-----
::

    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from avex import load_model

from utils.probing import compute_training_free_metrics, run_attention_probe, run_linear_probe
from utils.visualization import (
    plot_model_comparison,
    plot_model_comparison_static,
    plot_umap,
    plot_umap_static,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``config`` attribute.
    """
    parser = argparse.ArgumentParser(description="Train probes on giant otter embeddings.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def download_dataset(url: str, audio_dir: str) -> None:
    """Download and unzip the giant otter audio dataset if not already present.

    Parameters
    ----------
    url : str
        URL of the zip archive.
    audio_dir : str
        Target directory for extracted audio files.
    """
    audio_path = Path(audio_dir)
    if audio_path.exists() and any(audio_path.rglob("*.wav")):
        print(f"Dataset already present at {audio_dir}, skipping download.")
        return

    audio_path.mkdir(parents=True, exist_ok=True)
    zip_path = audio_path / "dataset.zip"
    print(f"Downloading dataset from {url} ...")
    urlretrieve(url, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(audio_path)
    zip_path.unlink()
    print(f"Dataset extracted to {audio_dir}.")


def collect_files_and_labels(audio_dir: str) -> tuple[list[Path], list[str]]:
    """Scan the audio directory and extract call-type labels from filenames.

    Giant otter filenames encode the call type as the first alphabetic token
    (e.g. ``bark_01_20190315.wav`` → label ``bark``).

    Parameters
    ----------
    audio_dir : str
        Root directory containing ``.wav`` files.

    Returns
    -------
    tuple[list[Path], list[str]]
        Parallel lists of file paths and corresponding string labels.
    """
    pattern = re.compile(r"^([a-zA-Z]+)")
    files, labels = [], []
    for wav in sorted(Path(audio_dir).rglob("*.wav")):
        m = pattern.match(wav.stem)
        if m:
            files.append(wav)
            labels.append(m.group(1).lower())
    return files, labels


def extract_embeddings(
    files: list[Path],
    model_name: str,
    sample_rate: int,
    window_seconds: float,
    pooling: str,
    device: str = "cpu",
) -> np.ndarray:
    """Extract mean-pooled embeddings for a list of audio files.

    Each file is padded or trimmed to ``window_seconds`` before the forward
    pass.  The raw feature output (shape ``(1, T, D)`` for BEATs or
    ``(1, C, H, W)`` for EfficientNet) is reduced to a 1-D vector via the
    chosen pooling strategy.

    Parameters
    ----------
    files : list[Path]
        Audio files to process.
    model_name : str
        Registered avex model identifier.
    sample_rate : int
        Target sample rate in Hz; audio is resampled if necessary.
    window_seconds : float
        Fixed clip length in seconds.
    pooling : str
        Pooling strategy: ``"mean"``, ``"max"``, or ``"cls"`` (EAT only).
    device : str
        PyTorch device string.

    Returns
    -------
    np.ndarray
        2D array of shape ``(n_files, embedding_dim)``.
    """
    model = load_model(model_name, return_features_only=True, device=device)
    model.eval()

    n_samples = int(window_seconds * sample_rate)
    all_embeddings: list[np.ndarray] = []

    for wav_path in files:
        audio, sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Pad or trim to fixed length
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        else:
            audio = audio[:n_samples]

        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor, padding_mask=None)

        # Pool to 1-D embedding
        if features.ndim == 4:
            # EfficientNet: (1, C, H, W) → global average pool
            emb = features.mean(dim=(2, 3)).squeeze(0)
        elif features.ndim == 3:
            # BEATs / EAT: (1, T, D)
            if pooling == "cls":
                emb = features[:, 0, :].squeeze(0)
            elif pooling == "max":
                emb = features.max(dim=1).values.squeeze(0)
            else:
                emb = features.mean(dim=1).squeeze(0)
        else:
            emb = features.squeeze(0)

        all_embeddings.append(emb.cpu().numpy())

    return np.stack(all_embeddings)


def extract_beats_all_layers_avg(
    files: list[Path],
    model_name: str,
    sample_rate: int,
    window_seconds: float,
    device: str = "cpu",
) -> np.ndarray:
    """Extract BEATs embeddings by averaging mean-pooled outputs across all layers.

    This uses forward hooks on the transformer encoder blocks and produces one
    embedding per audio clip, matching the dimensionality of a single layer.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_files, embedding_dim)`` with one averaged embedding
        per audio clip.
    """
    all_layers = extract_beats_all_layers(
        files=files,
        model_name=model_name,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        device=device,
    )
    return all_layers.mean(axis=0)


def extract_beats_all_layers(
    files: list[Path],
    model_name: str,
    sample_rate: int,
    window_seconds: float,
    device: str = "cpu",
) -> np.ndarray:
    """Extract mean-pooled BEATs embeddings for every encoder layer.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_layers, n_files, embedding_dim)`` containing
        mean-pooled embeddings for each encoder layer.
    """
    import librosa

    model = load_model(model_name, return_features_only=True, device=device)
    model.eval()

    n_samples = int(window_seconds * sample_rate)

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
        for wav_path in files:
            audio, sr = sf.read(str(wav_path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            if len(audio) < n_samples:
                audio = np.pad(audio, (0, n_samples - len(audio)))
            else:
                audio = audio[:n_samples]

            layer_store.clear()
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            _ = model(tensor, padding_mask=None)

            for i in range(n_layers):
                out = layer_store[i]
                pooled = out.view(-1, out.shape[-1]).mean(dim=0)
                per_layer_embs[i].append(pooled.cpu().numpy())

    for h in hooks:
        h.remove()
    del model

    return np.stack([np.stack(v) for v in per_layer_embs])


def main() -> None:
    """Run the full training pipeline for both avex models."""
    args = parse_args()
    cfg = load_config(args.config)

    download_dataset(cfg["dataset"]["url"], cfg["dataset"]["audio_dir"])
    files, labels = collect_files_and_labels(cfg["dataset"]["audio_dir"])
    print(f"Found {len(files)} audio files across {len(set(labels))} call types.")

    embeddings_dir = Path(cfg["output"]["embeddings_dir"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    accuracy_results: dict[str, float] = {}
    attention_accuracy: dict[str, float] = {}
    training_free_results: dict[str, dict] = {}

    beats_name = "esp_aves2_sl_beats_all"
    effnet_name = "esp_aves2_effnetb0_all"

    # --- BEATs last layer ----------------------------------------------------
    beats_last_path = embeddings_dir / "beats_embeddings.npy"
    if beats_last_path.exists():
        beats_last = np.load(beats_last_path)
    else:
        beats_last = extract_embeddings(
            files,
            model_name=beats_name,
            sample_rate=cfg["dataset"]["sample_rate"],
            window_seconds=cfg["dataset"]["window_seconds"],
            pooling="mean",
        )
        np.save(beats_last_path, beats_last)

    training_free_results[f"{beats_name} (last layer)"] = compute_training_free_metrics(
        beats_last, labels, random_state=int(cfg["probe"]["random_state"])
    )
    beats_last_probe = run_linear_probe(beats_last, labels, **cfg["probe"])
    accuracy_results[f"{beats_name} (last layer)"] = beats_last_probe["accuracy"]

    attn_probe_kw = dict(
        num_heads=8,
        num_attn_layers=2,
        epochs=50,
        test_size=float(cfg["probe"]["test_size"]),
        random_state=int(cfg["probe"]["random_state"]),
    )
    beats_last_attn = run_attention_probe(beats_last, labels, **attn_probe_kw)
    attention_accuracy[f"{beats_name} (last layer)"] = float(beats_last_attn["accuracy"])

    fig = plot_umap(
        beats_last,
        labels,
        title=f"Giant Otter — {beats_name} (last layer)",
        hover_text=[p.name for p in files],
    )
    fig.write_html(str(artifacts_dir / "umap_beats.html"), include_plotlyjs="cdn")
    fig_st = plot_umap_static(
        beats_last,
        labels,
        title=f"Giant Otter — {beats_name} (last layer)",
    )
    fig_st.savefig(str(artifacts_dir / "umap_beats_static.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_st)

    # --- BEATs all layers avg ------------------------------------------------
    beats_all_path = embeddings_dir / "beats_all_layers_embeddings.npy"
    beats_layers_dir = embeddings_dir / "beats_all_layers"
    if beats_all_path.exists():
        beats_all = np.load(beats_all_path)
    else:
        beats_layers = extract_beats_all_layers(
            files=files,
            model_name=beats_name,
            sample_rate=cfg["dataset"]["sample_rate"],
            window_seconds=cfg["dataset"]["window_seconds"],
        )
        beats_layers_dir.mkdir(parents=True, exist_ok=True)
        for i in range(beats_layers.shape[0]):
            np.save(beats_layers_dir / f"layer_{i:02d}.npy", beats_layers[i])
        beats_all = beats_layers.mean(axis=0)
        np.save(beats_all_path, beats_all)

    training_free_results[f"{beats_name} (all layers avg)"] = compute_training_free_metrics(
        beats_all, labels, random_state=int(cfg["probe"]["random_state"])
    )
    beats_all_probe = run_linear_probe(beats_all, labels, **cfg["probe"])
    accuracy_results[f"{beats_name} (all layers avg)"] = beats_all_probe["accuracy"]

    beats_all_attn = run_attention_probe(beats_all, labels, **attn_probe_kw)
    attention_accuracy[f"{beats_name} (all layers avg)"] = float(beats_all_attn["accuracy"])

    fig = plot_umap(
        beats_all,
        labels,
        title=f"Giant Otter — {beats_name} (all layers avg)",
        hover_text=[p.name for p in files],
    )
    fig.write_html(str(artifacts_dir / "umap_beats_all_layers.html"), include_plotlyjs="cdn")
    fig_st_all = plot_umap_static(
        beats_all,
        labels,
        title=f"Giant Otter — {beats_name} (all layers avg)",
    )
    fig_st_all.savefig(str(artifacts_dir / "umap_beats_all_layers_static.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_st_all)

    # --- EfficientNet --------------------------------------------------------
    effnet_path = embeddings_dir / "effnet_embeddings.npy"
    if effnet_path.exists():
        effnet = np.load(effnet_path)
    else:
        effnet = extract_embeddings(
            files,
            model_name=effnet_name,
            sample_rate=cfg["dataset"]["sample_rate"],
            window_seconds=cfg["dataset"]["window_seconds"],
            pooling="mean",
        )
        np.save(effnet_path, effnet)

    training_free_results[effnet_name] = compute_training_free_metrics(
        effnet, labels, random_state=int(cfg["probe"]["random_state"])
    )
    effnet_probe = run_linear_probe(effnet, labels, **cfg["probe"])
    accuracy_results[effnet_name] = effnet_probe["accuracy"]

    fig = plot_umap(
        effnet,
        labels,
        title=f"Giant Otter — {effnet_name}",
        hover_text=[p.name for p in files],
    )
    fig.write_html(str(artifacts_dir / "umap_effnet.html"), include_plotlyjs="cdn")
    fig_st_e = plot_umap_static(
        effnet,
        labels,
        title=f"Giant Otter — {effnet_name}",
    )
    fig_st_e.savefig(str(artifacts_dir / "umap_effnet_static.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_st_e)

    # Summary bar chart
    bar_fig = plot_model_comparison(accuracy_results, title="Giant Otter — model comparison")
    bar_fig.write_html(str(artifacts_dir / "model_comparison.html"), include_plotlyjs="cdn")
    bar_st = plot_model_comparison_static(accuracy_results, title="Giant Otter — model comparison")
    bar_st.savefig(str(artifacts_dir / "model_comparison_static.png"), dpi=150, bbox_inches="tight")
    plt.close(bar_st)

    beats_attn_cmp = {f"{k} (linear)": v for k, v in accuracy_results.items() if beats_name in k}
    beats_attn_cmp.update({f"{k} (attention)": v for k, v in attention_accuracy.items()})
    bar_attn = plot_model_comparison(
        beats_attn_cmp,
        title="Giant Otter — sl-BEATs: linear vs attention",
    )
    bar_attn.write_html(str(artifacts_dir / "model_comparison_beats_attn.html"), include_plotlyjs="cdn")
    bar_attn_st = plot_model_comparison_static(
        beats_attn_cmp,
        title="Giant Otter — sl-BEATs: linear vs attention",
    )
    bar_attn_st.savefig(str(artifacts_dir / "model_comparison_beats_attn_static.png"), dpi=150, bbox_inches="tight")
    plt.close(bar_attn_st)

    metrics = {
        "training_free": training_free_results,
        "linear_probe_accuracy": accuracy_results,
        "attention_probe_accuracy": attention_accuracy,
    }
    metrics_path = artifacts_dir / str(cfg["output"]["metrics_json"])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nArtifacts saved to {artifacts_dir}")
    print("\nLinear probe accuracy:")
    for name, acc in accuracy_results.items():
        print(f"  {name}: {acc:.1%}")
    print("\nAttention probe accuracy (BEATs only):")
    for name, acc in attention_accuracy.items():
        print(f"  {name}: {acc:.1%}")


if __name__ == "__main__":
    main()
