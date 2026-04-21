"""Visualization utilities for audio embeddings and probing results.

Provides interactive Plotly figures and static matplotlib figures used across
all avex-examples for exploring embedding spaces and tracking probing
performance.  Each interactive function has a ``*_static`` counterpart that
returns a ``matplotlib.figure.Figure`` for embedding in notebooks and web pages
without requiring JavaScript.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots


def plot_umap(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    title: str = "Embedding space (UMAP)",
    hover_text: list[str] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    color_discrete_map: dict[str, str] | None = None,
) -> go.Figure:
    """Reduce embeddings to 2D via UMAP and return an interactive Plotly scatter.

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape ``(n_samples, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample, used for colouring points.
    title : str
        Plot title.
    hover_text : list[str] | None
        Optional per-point hover annotations (e.g. filenames).
    n_neighbors : int
        UMAP ``n_neighbors`` parameter controlling local neighbourhood size.
    min_dist : float
        UMAP ``min_dist`` parameter controlling point compactness.
    random_state : int
        Random seed for reproducibility.
    color_discrete_map : dict[str, str] | None
        Optional mapping from label value to hex colour string.  When provided,
        each label is coloured with the given colour; unlisted labels fall back
        to Plotly's default palette.

    Returns
    -------
    go.Figure
        Interactive Plotly scatter figure.

    Examples
    --------
    >>> import numpy as np
    >>> emb = np.random.randn(20, 64)
    >>> lbl = ["a"] * 10 + ["b"] * 10
    >>> fig = plot_umap(emb, lbl, title="Test")
    >>> fig.layout.title.text
    'Test'
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    coords = reducer.fit_transform(embeddings)

    df_dict: dict = {"x": coords[:, 0], "y": coords[:, 1], "label": list(labels)}
    if hover_text is not None:
        df_dict["hover"] = hover_text

    fig = px.scatter(
        df_dict,
        x="x",
        y="y",
        color="label",
        color_discrete_map=color_discrete_map,
        hover_name="hover" if hover_text is not None else None,
        title=title,
        labels={"x": "UMAP 1", "y": "UMAP 2"},
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    return fig


def plot_umap_grid(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    layer_indices: list[int],
    layer_embeddings: list[np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> go.Figure:
    """Plot UMAP projections for multiple embedding layers side by side.

    Parameters
    ----------
    embeddings : np.ndarray
        Final-layer embeddings, shape ``(n_samples, embedding_dim)``. Used as
        reference for the last panel.
    labels : list[str] | np.ndarray
        Class label for each sample.
    layer_indices : list[int]
        Transformer layer indices corresponding to ``layer_embeddings``.
    layer_embeddings : list[np.ndarray]
        Per-layer embeddings; each entry has shape ``(n_samples, embedding_dim)``.
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    go.Figure
        Plotly figure with one subplot per layer.
    """
    n_panels = len(layer_indices)
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=[f"Layer {i}" for i in layer_indices])

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    unique_labels = sorted(set(labels))
    color_map = {lbl: px.colors.qualitative.Plotly[i % 10] for i, lbl in enumerate(unique_labels)}

    for col, (_layer_idx, layer_emb) in enumerate(zip(layer_indices, layer_embeddings, strict=True), start=1):
        coords = reducer.fit_transform(layer_emb)
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    marker={"size": 5, "color": color_map[lbl], "opacity": 0.8},
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(title="Embedding space across BEATs layers", height=450)
    return fig


def plot_umap_grid_static(
    labels: list[str] | np.ndarray,
    layer_indices: list[int],
    layer_embeddings: list[np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Static matplotlib version of :func:`plot_umap_grid`.

    Parameters
    ----------
    labels : list[str] | np.ndarray
        Class label for each sample.
    layer_indices : list[int]
        Transformer layer indices corresponding to ``layer_embeddings``.
    layer_embeddings : list[np.ndarray]
        Per-layer embeddings; each entry has shape ``(n_samples, embedding_dim)``.
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        Random seed for reproducibility.
    figsize : tuple[int, int] | None
        Figure size; defaults to ``(5 * n_panels, 4)``.

    Returns
    -------
    plt.Figure
        Static matplotlib figure with one subplot per layer.
    """
    n_panels = len(layer_indices)
    if figsize is None:
        figsize = (5 * n_panels, 4)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, layer_idx, layer_emb in zip(axes, layer_indices, layer_embeddings, strict=False):
        coords = reducer.fit_transform(layer_emb)
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=color_map[lbl],
                label=lbl,
                s=18,
                alpha=0.8,
                linewidths=0,
            )
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize="x-small",
        frameon=False,
    )
    fig.suptitle("Embedding space across BEATs layers", fontsize=12)
    fig.tight_layout()
    return fig


def plot_layer_curve(
    layer_accuracies: list[float],
    dataset_name: str = "",
    model_name: str = "BEATs",
) -> go.Figure:
    """Plot linear probe accuracy vs transformer layer index.

    Parameters
    ----------
    layer_accuracies : list[float]
        Probe accuracy at each layer, from layer 0 (earliest) to last.
    dataset_name : str
        Dataset label shown in the plot title.
    model_name : str
        Model name shown in the plot title.

    Returns
    -------
    go.Figure
        Plotly line chart with one point per layer.

    Examples
    --------
    >>> accs = [0.3, 0.5, 0.6, 0.65, 0.7]
    >>> fig = plot_layer_curve(accs, "test_data")
    >>> len(fig.data) == 1
    True
    """
    layers = list(range(len(layer_accuracies)))
    fig = go.Figure(
        go.Scatter(
            x=layers,
            y=layer_accuracies,
            mode="lines+markers",
            marker={"size": 8},
            line={"width": 2},
        )
    )
    fig.update_layout(
        title=f"{model_name} probing accuracy by layer — {dataset_name}",
        xaxis_title="Transformer layer",
        yaxis_title="Linear probe accuracy",
        yaxis={"range": [0, 1]},
    )
    return fig


def plot_model_comparison(
    results: dict[str, float],
    title: str = "Model comparison",
) -> go.Figure:
    """Bar chart comparing probe accuracy across models or pooling strategies.

    Parameters
    ----------
    results : dict[str, float]
        Mapping from model/strategy name to accuracy value.
    title : str
        Plot title.

    Returns
    -------
    go.Figure
        Plotly bar chart.

    Examples
    --------
    >>> res = {"BEATs last": 0.82, "BEATs all": 0.85, "EfficientNet": 0.78}
    >>> fig = plot_model_comparison(res)
    >>> len(fig.data) == 1
    True
    """
    fig = go.Figure(
        go.Bar(
            x=list(results.keys()),
            y=list(results.values()),
            text=[f"{v:.1%}" for v in results.values()],
            textposition="auto",
        )
    )
    fig.update_layout(title=title, yaxis={"title": "Accuracy", "range": [0, 1]})
    return fig


# ---------------------------------------------------------------------------
# Static (matplotlib) equivalents — render without JavaScript in notebooks
# and web pages.
# ---------------------------------------------------------------------------


def plot_umap_static(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    title: str = "Embedding space (UMAP)",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
    color_map: dict[str, str] | None = None,
    legend_below: bool = False,
) -> plt.Figure:
    """Reduce embeddings to 2D via UMAP and return a static matplotlib scatter.

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape ``(n_samples, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample, used for colouring points.
    title : str
        Plot title.
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        Random seed for reproducibility.
    figsize : tuple[int, int]
        Matplotlib figure size ``(width, height)`` in inches.
    color_map : dict[str, str] | None
        Optional explicit label → colour mapping.  When ``None`` the ``tab20``
        (or ``hsv`` for >20 classes) colourmap is used.
    legend_below : bool
        When ``True`` the legend is placed below the scatter in multiple
        columns instead of to the right.  Useful when there are many labels.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    if color_map is None:
        cmap = plt.cm.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
        color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
    else:
        # Fall back to grey for any label not in the provided map
        unlisted = [lbl for lbl in unique_labels if lbl not in color_map]
        if unlisted:
            color_map = dict(color_map)
            for lbl in unlisted:
                color_map[lbl] = "#aaaaaa"

    fig, ax = plt.subplots(figsize=figsize)
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color_map[lbl]], label=lbl, s=18, alpha=0.8, linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    if legend_below:
        ncol = max(1, min(8, int(np.ceil(n_classes / 12))))
        n_rows_legend = int(np.ceil(n_classes / ncol))
        legend_height_in = n_rows_legend * 0.18 + 0.3
        w, h = fig.get_size_inches()
        new_h = h + legend_height_in
        fig.set_size_inches(w, new_h)
        ax.legend(
            bbox_to_anchor=(0.5, 0),
            loc="upper center",
            ncol=ncol,
            fontsize="x-small",
            frameon=False,
            bbox_transform=fig.transFigure,
        )
        fig.tight_layout(rect=[0, legend_height_in / new_h, 1, 1])
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", frameon=False)
        fig.tight_layout()
    return fig


def plot_model_comparison_static(
    results: dict[str, float | None],
    title: str = "Model comparison",
    figsize: tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Bar chart comparing probe accuracy across models — static matplotlib version.

    Parameters
    ----------
    results : dict[str, float | None]
        Mapping from model/strategy name to accuracy value. Missing values may be
        ``None`` and will be rendered as empty (non-annotated) bars.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    names = list(results.keys())
    raw_values = list(results.values())
    values: list[float] = []
    for v in raw_values:
        if v is None:
            values.append(np.nan)
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(names)), values, color=plt.cm.tab10(np.linspace(0, 0.9, len(names))))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize="small")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for bar, val in zip(bars, values, strict=True):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize="small",
            )
    fig.tight_layout()
    return fig


def plot_layer_curve_static(
    layer_accuracies: list[float],
    dataset_name: str = "",
    model_name: str = "BEATs",
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot linear probe accuracy vs transformer layer index — static matplotlib version.

    Parameters
    ----------
    layer_accuracies : list[float]
        Probe accuracy at each layer, from layer 0 (earliest) to last.
    dataset_name : str
        Dataset label shown in the plot title.
    model_name : str
        Model name shown in the plot title.
    figsize : tuple[int, int]
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    layers = list(range(len(layer_accuracies)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, layer_accuracies, marker="o", linewidth=2, markersize=7)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} probing accuracy by layer — {dataset_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_heatmap_static(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion matrix",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Normalised confusion matrix heatmap — static matplotlib version.

    Parameters
    ----------
    cm : np.ndarray
        Integer confusion matrix of shape ``(n_classes, n_classes)``.
    classes : list[str]
        Class names for the axes.
    title : str
        Plot title.
    figsize : tuple[int, int] | None
        Matplotlib figure size; defaults to ``(n_classes, n_classes)`` clamped
        to a minimum of ``(6, 5)``.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    n = len(classes)
    if figsize is None:
        figsize = (max(6, n), max(5, n))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize="small")
    ax.set_yticklabels(classes, fontsize="small")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize="x-small",
                color="white" if val > thresh else "black",
            )

    fig.tight_layout()
    return fig


def per_species_probe_heatmap_static(
    probe_df: pd.DataFrame,
    *,
    value_col: str = "balanced_accuracy",
    row_label: str = "species",
    col_label: str = "model",
    title: str = "Per-species probe — balanced accuracy (static)",
    figsize: tuple[float, float] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Heatmap of per-species scores (e.g. balanced accuracy) vs embedding model.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Long-form table with at least species, model, and a numeric score column.
    value_col : str
        Column name for the value displayed in each cell.
    row_label : str
        Column used as heatmap rows (typically ``\"species\"``).
    col_label : str
        Column used as heatmap columns (typically ``\"model\"``).
    title : str
        Plot title.
    figsize : tuple[float, float] | None
        Figure size in inches; defaults from table shape.
    vmin : float
        Colour scale minimum.
    vmax : float
        Colour scale maximum.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    pivot = probe_df.pivot_table(index=row_label, columns=col_label, values=value_col, aggfunc="mean")
    n_rows, n_cols = pivot.shape
    if figsize is None:
        figsize = (max(8, 0.35 * n_cols + 4), max(6, 0.22 * n_rows + 2))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value_col.replace("_", " ").title())

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right", fontsize="small")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.tolist(), fontsize="small")
    ax.set_title(title)
    ax.set_xlabel(col_label.replace("_", " ").title())
    ax.set_ylabel(row_label.replace("_", " ").title())
    fig.tight_layout()
    return fig


def pivot_grouped_bar_static(
    df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    title: str,
    ylabel: str = "",
    ylim_top: float = 1.05,
    rot: float = 18,
    figsize: tuple[float, float] = (10, 5),
    hline_y: float | None = None,
    hline_label: str = "",
) -> plt.Figure:
    """Grouped bar chart from a long table via ``pivot(index, columns, values)``.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form data (e.g. model × protocol × score).
    index : str
        Column for categorical x-axis groups.
    columns : str
        Column whose unique values become bar series (legend).
    values : str
        Numeric height of each bar.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    ylim_top : float
        Upper y-axis limit.
    rot : float
        X tick label rotation (degrees).
    figsize : tuple[float, float]
        Figure size in inches.
    hline_y : float | None
        Optional horizontal reference line (e.g. chance).
    hline_label : str
        Label for the reference line in the legend.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    pv = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=figsize)
    pv.plot(kind="bar", ax=ax, width=0.8, rot=rot)
    ax.set_title(title)
    ax.set_ylabel(ylabel or values.replace("_", " ").title())
    ax.set_xlabel(index.replace("_", " ").title())
    ax.legend(title=columns.replace("_", " ").title(), fontsize="small")
    ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", alpha=0.25)
    if hline_y is not None:
        ax.axhline(hline_y, color="grey", linestyle="--", linewidth=1)
        if hline_label:
            ax.text(ax.get_xlim()[1] * 0.98, hline_y + 0.02, hline_label, ha="right", fontsize="small", color="grey")
    fig.tight_layout()
    return fig


def horizontal_counts_bar_static(
    counts_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    figsize: tuple[float, float] = (8, 10),
) -> plt.Figure:
    """Horizontal bar chart of class counts (static matplotlib).

    Parameters
    ----------
    counts_df : pd.DataFrame
        Data frame with count and label columns.
    x_col : str
        Column for bar lengths (counts).
    y_col : str
        Column for categorical y labels.
    title : str
        Plot title.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    plt.Figure
        Static matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_vals = counts_df[y_col].tolist()
    x_vals = counts_df[x_col].to_numpy()
    ax.barh(range(len(y_vals)), x_vals, color="#2a9d8f")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals, fontsize="small")
    ax.invert_yaxis()
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig
