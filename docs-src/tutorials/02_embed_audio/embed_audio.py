"""Batch audio embedding extraction using two avex models.

Scans an input directory for audio files, shards each file into fixed-length
overlapping windows, extracts embeddings with ``esp_aves2_sl_beats_all`` and
``esp_aves2_effnetb0_all``, and saves one ``.pt`` tensor file per audio file
per model in the configured output directory.

Usage
-----
::

    python embed_audio.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from avex import load_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a ``config`` attribute.
    """
    parser = argparse.ArgumentParser(description="Batch audio embedding extraction.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def shard_audio(audio: np.ndarray, sample_rate: int, window_seconds: float, hop_seconds: float) -> list[np.ndarray]:
    """Split a 1-D audio array into fixed-length overlapping windows.

    Incomplete trailing windows are zero-padded to ``window_seconds``.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio array.
    sample_rate : int
        Sample rate in Hz.
    window_seconds : float
        Length of each window in seconds.
    hop_seconds : float
        Hop between consecutive windows in seconds.

    Returns
    -------
    list[np.ndarray]
        List of fixed-length audio windows.
    """
    win_len = int(window_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)

    windows: list[np.ndarray] = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + win_len]
        if len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)))
        windows.append(chunk)
        start += hop_len

    return windows


def embed_file(
    wav_path: Path,
    model: torch.nn.Module,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    device: str,
) -> torch.Tensor:
    """Extract mean-pooled window embeddings for a single audio file.

    Parameters
    ----------
    wav_path : Path
        Path to the input ``.wav`` file.
    model : torch.nn.Module
        Loaded avex model in feature-extraction mode.
    sample_rate : int
        Target sample rate in Hz.
    window_seconds : float
        Window length in seconds.
    hop_seconds : float
        Hop size in seconds.
    device : str
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_windows, embedding_dim)`` containing one
        mean-pooled embedding per window.
    """
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    windows = shard_audio(audio, sample_rate, window_seconds, hop_seconds)
    embeddings: list[torch.Tensor] = []

    model.eval()
    for window in windows:
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor, padding_mask=None)

        # Pool spatial / temporal dimensions to a 1-D vector
        if features.ndim == 4:
            emb = features.mean(dim=(2, 3)).squeeze(0)
        elif features.ndim == 3:
            emb = features.mean(dim=1).squeeze(0)
        else:
            emb = features.squeeze(0)

        embeddings.append(emb.cpu())

    return torch.stack(embeddings)


def main() -> None:
    """Extract and save embeddings for all audio files in the input directory.

    Raises
    ------
    FileNotFoundError
        If no ``.wav`` or ``.flac`` files are found in the configured input directory.
    """
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = Path(cfg["audio"]["input_dir"])
    output_dir = Path(cfg["output"]["embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.rglob("*.wav")) + sorted(input_dir.rglob("*.flac"))
    if not audio_files:
        raise FileNotFoundError(f"No .wav or .flac files found in {input_dir}")

    print(f"Found {len(audio_files)} audio files.")

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        print(f"\nLoading model: {model_name}")
        model = load_model(model_name, return_features_only=True, device="cpu")
        model_out_dir = output_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in audio_files:
            out_path = model_out_dir / (wav_path.stem + ".pt")
            if out_path.exists():
                print(f"  Skipping {wav_path.name} (already extracted).")
                continue

            print(f"  Processing {wav_path.name} ...")
            emb = embed_file(
                wav_path,
                model=model,
                sample_rate=cfg["audio"]["sample_rate"],
                window_seconds=cfg["audio"]["window_seconds"],
                hop_seconds=cfg["audio"]["hop_seconds"],
                device="cpu",
            )
            torch.save(emb, str(out_path))

        print(f"Embeddings saved to {model_out_dir}")


if __name__ == "__main__":
    main()
