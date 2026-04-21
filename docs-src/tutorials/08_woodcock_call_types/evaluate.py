"""Evaluate and visualize results for Eurasian woodcock call type classification.

This script is intentionally lightweight: it reads the artifacts created by `train.py`
(metadata, embeddings, metrics) and produces a small set of saved figures.

Usage
-----
```bash
uv run python examples/08_woodcock_call_types/evaluate.py --config examples/08_woodcock_call_types/config.yaml
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import yaml

from utils.visualization import plot_umap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate woodcock artifacts.")
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


def main() -> None:
    """Generate HTML figures from saved artifacts."""
    args = parse_args()
    repo_root = find_repo_root(pathlib.Path(__file__).resolve())
    example_dir = repo_root / "examples" / "08_woodcock_call_types"

    cfg_path = (
        (repo_root / args.config).resolve()
        if not pathlib.Path(args.config).is_absolute()
        else pathlib.Path(args.config)
    )
    cfg = load_config(cfg_path)

    meta_path = example_dir / str(cfg["dataset"]["metadata_csv"])
    df = pd.read_csv(meta_path)

    emb_dir = example_dir / str(cfg["embeddings"]["dir"])
    beats = np.load(emb_dir / str(cfg["embeddings"]["beats_file"]))
    beats_all = np.load(emb_dir / "beats_all_layers_embeddings.npy")
    effnet = np.load(emb_dir / str(cfg["embeddings"]["effnet_file"]))

    out_dir = example_dir / str(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # UMAPs (saved as HTML)
    labels = df["call_type"].astype(str).tolist()
    hover = [
        f"{pathlib.Path(p).name}<br>Type: {ct}<br>Deploy: {dep}"
        for p, ct, dep in zip(df["path"], df["call_type"], df["deploy_id"], strict=False)
    ]

    fig_beats = plot_umap(
        beats,
        labels=labels,
        title=f"UMAP — {cfg['models']['beats']['name']}<br><sup>colour = call type</sup>",
        hover_text=hover,
    )
    fig_beats.write_html(out_dir / "umap_beats.html", include_plotlyjs="cdn")

    fig_beats_all = plot_umap(
        beats_all,
        labels=labels,
        title=f"UMAP — {cfg['models']['beats']['name']} (all layers avg)<br><sup>colour = call type</sup>",
        hover_text=hover,
    )
    fig_beats_all.write_html(out_dir / "umap_beats_all_layers.html", include_plotlyjs="cdn")

    fig_effnet = plot_umap(
        effnet,
        labels=labels,
        title=f"UMAP — {cfg['models']['effnet']['name']}<br><sup>colour = call type</sup>",
        hover_text=hover,
    )
    fig_effnet.write_html(out_dir / "umap_effnet.html", include_plotlyjs="cdn")

    # Metrics bar chart (if present)
    metrics_path = out_dir / str(cfg["output"]["metrics_json"])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = [
            {
                "model": "BEATs (last layer)",
                "protocol": "Random split",
                "balanced_accuracy": metrics["beats_last_layer"]["random_split"]["balanced_accuracy"],
            },
            {
                "model": "BEATs (last layer)",
                "protocol": "LODO",
                "balanced_accuracy": metrics["beats_last_layer"]["lodo"]["mean_balanced_accuracy"],
            },
            {
                "model": "BEATs (all layers avg)",
                "protocol": "Random split",
                "balanced_accuracy": metrics["beats_all_layers_avg"]["random_split"]["balanced_accuracy"],
            },
            {
                "model": "BEATs (all layers avg)",
                "protocol": "LODO",
                "balanced_accuracy": metrics["beats_all_layers_avg"]["lodo"]["mean_balanced_accuracy"],
            },
            {
                "model": "EffNet",
                "protocol": "Random split",
                "balanced_accuracy": metrics["effnet"]["random_split"]["balanced_accuracy"],
            },
            {
                "model": "EffNet",
                "protocol": "LODO",
                "balanced_accuracy": metrics["effnet"]["lodo"]["mean_balanced_accuracy"],
            },
        ]
        mdf = pd.DataFrame(rows)
        fig = px.bar(
            mdf,
            x="model",
            y="balanced_accuracy",
            color="protocol",
            barmode="group",
            title="Woodcock call-type classification (balanced accuracy)",
            labels={"balanced_accuracy": "Balanced accuracy"},
        )
        fig.write_html(out_dir / "balanced_accuracy.html", include_plotlyjs="cdn")

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
