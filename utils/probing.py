"""Linear, MLP, and attention probe utilities for evaluating audio embeddings.

Provides lightweight probing classifiers used across all avex-examples
to measure how well embeddings encode task-relevant information.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def run_linear_probe(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> dict:
    """Train a logistic regression probe and return classification metrics.

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape ``(n_samples, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample.
    test_size : float
        Fraction of samples held out for evaluation.
    random_state : int
        Random seed for reproducibility.
    max_iter : int
        Maximum iterations for the logistic regression solver.

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``confusion_matrix``,
        ``classes``, ``y_test``, and ``y_pred``.

    Examples
    --------
    >>> import numpy as np
    >>> emb = np.random.randn(100, 64)
    >>> lbl = ["a"] * 50 + ["b"] * 50
    >>> result = run_linear_probe(emb, lbl)
    >>> 0.0 <= result["accuracy"] <= 1.0
    True
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return {
            "accuracy": None,
            "confusion_matrix": np.zeros((classes.size, classes.size), dtype=int),
            "classes": le.classes_.tolist(),
            "y_test": np.array([], dtype=int),
            "y_pred": np.array([], dtype=int),
        }

    # Stratified split requires enough samples per class for both splits.
    stratify: np.ndarray | None = None
    if counts.min(initial=0) >= 2:
        stratify = y

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # With very small datasets, an unstratified split can produce a train or test
    # set containing a single class, which would make the probe ill-defined.
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {
            "accuracy": None,
            "confusion_matrix": np.zeros((2, 2), dtype=int),
            "classes": le.classes_.tolist(),
            "y_test": y_test,
            "y_pred": np.array([], dtype=int),
        }

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    try:
        clf.fit(x_train, y_train)
    except ValueError:
        return {
            "accuracy": None,
            "confusion_matrix": np.zeros((2, 2), dtype=int),
            "classes": le.classes_.tolist(),
            "y_test": y_test,
            "y_pred": np.array([], dtype=int),
        }
    y_pred = clf.predict(x_test)

    all_labels = np.arange(len(le.classes_), dtype=int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=all_labels),
        "classes": le.classes_.tolist(),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def compute_training_free_metrics(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    random_state: int = 42,
) -> dict:
    """Compute training-free embedding quality metrics (NMI, ARI, R-AUC).

    These metrics quantify how well the embedding space reflects the class
    structure without fitting any classifier. They are computed after
    embedding extraction and before any supervised probe.

    - **NMI** (Normalized Mutual Information): compares k-means cluster
      assignments to ground-truth labels; 0 = random, 1 = perfect.
    - **ARI** (Adjusted Rand Index): cluster–label agreement adjusted for
      chance; 0 = random, 1 = perfect.
    - **R-AUC** (Retrieval AUC / mAP): for each query embedding, ranks all
      other embeddings by cosine distance and computes average precision
      (fraction of same-class items retrieved before each hit). Averaged
      across all queries; equivalent to mean average precision (mAP).

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape ``(n_samples, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample.
    random_state : int
        Random seed for k-means reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``nmi``, ``ari``, and ``r_auc``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> emb = np.vstack([rng.normal([i * 5, 0], 1, (30, 2)) for i in range(3)])
    >>> lbl = ["a"] * 30 + ["b"] * 30 + ["c"] * 30
    >>> m = compute_training_free_metrics(emb, lbl)
    >>> m["nmi"] > 0.5
    True
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = int(le.classes_.size)

    if n_classes < 2 or len(embeddings) < n_classes:
        return {"nmi": None, "ari": None, "r_auc": None}

    # --- NMI and ARI via k-means clustering ----------------------------------
    kmeans = KMeans(n_clusters=n_classes, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    nmi = float(normalized_mutual_info_score(y, cluster_labels))
    ari = float(adjusted_rand_score(y, cluster_labels))

    # --- R-AUC (mean average precision for nearest-neighbour retrieval) ------
    dists = cosine_distances(embeddings)
    aps: list[float] = []
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        d = dists[i][mask]
        relevant = (y[mask] == y[i]).astype(int)

        if relevant.sum() == 0:
            continue

        order = np.argsort(d)
        relevant_sorted = relevant[order]

        n_retrieved = 0
        precisions: list[float] = []
        for k, rel in enumerate(relevant_sorted, 1):
            if rel:
                n_retrieved += 1
                precisions.append(n_retrieved / k)

        aps.append(float(np.mean(precisions)) if precisions else 0.0)

    r_auc = float(np.mean(aps)) if aps else None

    return {"nmi": nmi, "ari": ari, "r_auc": r_auc}


class _MLPProbe(nn.Module):
    """Small two-layer MLP classification head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Number of neurons in the hidden layer.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        """Initialise the MLP probe layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        return self.net(x)


def run_mlp_probe(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cpu",
) -> dict:
    """Train a small MLP probe and return per-epoch accuracy history.

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape ``(n_samples, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample.
    hidden_dim : int
        Number of neurons in the hidden layer.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    test_size : float
        Fraction of samples held out for evaluation.
    random_state : int
        Random seed for reproducibility.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict
        Dictionary with keys ``train_accuracy``, ``test_accuracy`` (lists,
        one value per epoch), ``classes``, and ``final_accuracy``.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return {
            "train_accuracy": [],
            "test_accuracy": [],
            "final_accuracy": None,
            "classes": le.classes_.tolist(),
        }

    stratify: np.ndarray | None = None
    if counts.min(initial=0) >= 2:
        stratify = y

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    x_tr = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    x_te = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    num_classes = len(le.classes_)
    model = _MLPProbe(x_tr.shape[1], hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_acc_hist: list[float] = []
    test_acc_hist: list[float] = []

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = (model(x_tr).argmax(1) == y_tr).float().mean().item()
            test_acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

    return {
        "train_accuracy": train_acc_hist,
        "test_accuracy": test_acc_hist,
        "final_accuracy": test_acc_hist[-1],
        "classes": le.classes_.tolist(),
    }


def run_attention_probe(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    num_heads: int = 8,
    num_attn_layers: int = 2,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cpu",
) -> dict:
    """Train an avex AttentionProbe on pre-extracted embeddings.

    Uses :class:`avex.models.probes.AttentionProbe` in *feature mode*
    (backbone frozen / not used).  Accepts both mean-pooled 2-D embeddings
    ``(n_samples, embedding_dim)`` and raw token sequences ``(n_samples,
    seq_len, embedding_dim)``.  For 2-D inputs a sequence dimension of 1 is
    added so that the attention mechanism still operates on the feature axis.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array ``(n_samples, embedding_dim)`` **or** 3-D array
        ``(n_samples, seq_len, embedding_dim)``.
    labels : list[str] | np.ndarray
        Class label for each sample.
    num_heads : int
        Number of attention heads (must divide ``embedding_dim``).  Reduced
        automatically if ``embedding_dim`` is not divisible.
    num_attn_layers : int
        Number of stacked multi-head-attention blocks.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    test_size : float
        Fraction of samples held out for evaluation.
    random_state : int
        Random seed for reproducibility.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``confusion_matrix``, ``classes``,
        ``y_test``, and ``y_pred``.

    Examples
    --------
    >>> import numpy as np
    >>> emb = np.random.randn(100, 64)
    >>> lbl = ["a"] * 50 + ["b"] * 50
    >>> result = run_attention_probe(emb, lbl, num_heads=8, epochs=5)
    >>> 0.0 <= result["accuracy"] <= 1.0
    True
    """
    from avex.models.probes import AttentionProbe

    le = LabelEncoder()
    y = le.fit_transform(labels)

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return {
            "accuracy": None,
            "confusion_matrix": np.zeros((classes.size, classes.size), dtype=int),
            "classes": le.classes_.tolist(),
            "y_test": np.array([], dtype=int),
            "y_pred": np.array([], dtype=int),
        }

    stratify: np.ndarray | None = None
    if counts.min(initial=0) >= 2:
        stratify = y

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {
            "accuracy": None,
            "confusion_matrix": np.zeros((2, 2), dtype=int),
            "classes": le.classes_.tolist(),
            "y_test": y_test,
            "y_pred": np.array([], dtype=int),
        }

    # For 2-D mean-pooled embeddings add a sequence dimension → (n, 1, D).
    # For 3-D token sequences use as-is → (n, T, D).
    if embeddings.ndim == 2:
        x_train_3d = x_train[:, np.newaxis, :]
        x_test_3d = x_test[:, np.newaxis, :]
    else:
        x_train_3d = x_train
        x_test_3d = x_test

    feat_dim = int(x_train_3d.shape[-1])
    num_classes = int(le.classes_.size)

    # Ensure num_heads divides feat_dim.
    while feat_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    probe = AttentionProbe(
        base_model=None,
        layers=[],
        num_classes=num_classes,
        device=device,
        feature_mode=True,
        input_dim=feat_dim,
        num_heads=num_heads,
        num_layers=num_attn_layers,
    )

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_tr = torch.tensor(x_train_3d, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    x_te = torch.tensor(x_test_3d, dtype=torch.float32).to(device)

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(probe(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        y_pred = probe(x_te).argmax(1).cpu().numpy()

    all_labels = np.arange(num_classes, dtype=int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=all_labels),
        "classes": le.classes_.tolist(),
        "y_test": y_test,
        "y_pred": y_pred,
    }
