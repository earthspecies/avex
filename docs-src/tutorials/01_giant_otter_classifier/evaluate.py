"""Evaluate saved giant otter embeddings with a trained linear probe.

Loads pre-extracted ``.npy`` embedding files produced by ``train.py`` and
re-runs the linear probe evaluation, printing per-class accuracy and saving
a confusion matrix figure.

Usage
-----
::

    python evaluate.py --config config.yaml --model esp_aves2_sl_beats_all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.figure_factory as ff
import yaml

from utils.probing import run_linear_probe


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``config`` and ``model`` attributes.
    """
    parser = argparse.ArgumentParser(description="Evaluate giant otter probe.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="esp_aves2_sl_beats_all", help="Model name to evaluate.")
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


def main() -> None:
    """Load embeddings and run evaluation for one model.

    Raises
    ------
    FileNotFoundError
        If the embeddings ``.npy`` file for the requested model does not exist.
    """
    args = parse_args()
    cfg = load_config(args.config)

    emb_path = Path(cfg["output"]["embeddings_dir"]) / f"{args.model}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_path}. Run train.py first.")

    emb = np.load(str(emb_path))
    _, labels = collect_files_and_labels(cfg["dataset"]["audio_dir"])
    result = run_linear_probe(emb, labels, **cfg["probe"])

    print(f"Model: {args.model}")
    print(f"Accuracy: {result['accuracy']:.1%}")

    artifacts_dir = Path(cfg["output"].get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cm = result["confusion_matrix"].astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig = ff.create_annotated_heatmap(
        z=cm_norm.tolist(),
        x=result["classes"],
        y=result["classes"],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=f"Confusion matrix — {args.model}",
        xaxis_title="Predicted",
        yaxis_title="True",
    )
    out_path = artifacts_dir / f"confusion_{args.model}.html"
    fig.write_html(str(out_path))
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
