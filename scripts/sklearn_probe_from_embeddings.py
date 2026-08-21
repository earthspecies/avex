"""Re-probe AVEX-extracted embeddings with a converged linear classifier.

WHY THIS EXISTS
  avex's probe reports substantially lower accuracy than the same linear model
  fitted to convergence. Measured on dog_classification, identical embeddings and
  identical splits (384 train / 139 test), only the probe differing:

      avex probe (900 epochs, lr 1e-4, wd 0.1) : 0.662
      this script (lbfgs, converged)           : 0.835   (+17.3 pts)

  This is NOT a more expressive model. Multinomial logistic regression is
  W.x + b with softmax and cross-entropy -- the same function class, the same
  parameter count, as nn.Linear(768, C) + CrossEntropyLoss. The gap is purely
  optimisation and regularisation:
    * lbfgs runs to convergence; a fixed epoch budget may not
    * weight_decay 0.1 is very strong for a linear head
    * feature standardisation conditions the problem (avex does not do it)
  So the torch probe should be able to reach these numbers too; this script
  measures what the representation actually supports, independent of that.

WHAT IT READS
  The embeddings avex already wrote during evaluation -- nothing is recomputed,
  so splits, labels and features are exactly avex's:
      <results_dir>/<dataset>_<model>/embedding_{train,val,test}_<layer>.h5

  Regularisation strength C is chosen on the VALIDATION split (not test), then
  the chosen model is scored once on test.

USAGE
  uv run python scripts/sklearn_probe_from_embeddings.py \
      --results-dir evaluation_results/icassp2026_pupujepa_bigrun_probe

  # compare several models side by side
  uv run python scripts/sklearn_probe_from_embeddings.py \
      --results-dir evaluation_results/icassp2026_pupujepa_bigrun_probe \
      --results-dir evaluation_results/icassp2026_eat_ssl_all_probe
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def _load_split(ds_dir: Path, split: str):
    """Returns (X, y, multi_label) or None when that split is absent."""
    files = sorted(ds_dir.glob(f"embedding_{split}_*.h5"))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        emb_key = next(k for k in h.keys() if k.startswith("embeddings"))
        X = h[emb_key][:]
        y = h["labels"][:]
    X = X.reshape(len(X), -1)
    if y.ndim == 2 and y.shape[1] > 1:
        multi = bool((y.sum(axis=1) > 1).any())
        y = y.argmax(axis=1)          # single-label one-hot -> class index
    else:
        multi = False
        y = y.ravel()
    return X, y, multi


def probe_dataset(ds_dir: Path, cs, max_iter: int, seed: int):
    tr = _load_split(ds_dir, "train")
    te = _load_split(ds_dir, "test")
    if tr is None or te is None:
        return None
    Xtr, ytr, multi_tr = tr
    Xte, yte, multi_te = te
    if multi_tr or multi_te:
        # Multi-label (detection) needs mAP over per-class scores, not accuracy.
        return {"dataset": ds_dir.name, "skipped": "multi-label"}

    va = _load_split(ds_dir, "val")
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    # Pick C on validation; fall back to C=1.0 when there is no val split.
    best_c = 1.0
    if va is not None and len(cs) > 1:
        Xva, yva, _ = va
        Zva = sc.transform(Xva)
        best_acc = -1.0
        for c in cs:
            m = LogisticRegression(C=c, max_iter=max_iter, random_state=seed).fit(Ztr, ytr)
            a = m.score(Zva, yva)
            if a > best_acc:
                best_acc, best_c = a, c

    clf = LogisticRegression(C=best_c, max_iter=max_iter, random_state=seed).fit(Ztr, ytr)
    pred = clf.predict(Zte)
    return {
        "dataset": ds_dir.name,
        "n_train": len(ytr),
        "n_test": len(yte),
        "n_classes": int(len(np.unique(ytr))),
        "C": best_c,
        "accuracy": float((pred == yte).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", action="append", required=True,
                   help="an avex evaluation_results/<run> directory (repeatable)")
    p.add_argument("--out", default="sklearn_probe_results.csv")
    p.add_argument("--C", default="0.01,0.1,1.0,10.0",
                   help="regularisation grid, selected on val")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cs = [float(c) for c in args.C.split(",")]
    rows = []
    for rd in args.results_dir:
        root = Path(rd)
        run = root.name
        ds_dirs = sorted(d for d in root.iterdir()
                         if d.is_dir() and list(d.glob("embedding_test_*.h5")))
        if not ds_dirs:
            print(f"[skip] no embeddings under {root}")
            continue
        print(f"\n=== {run} ===")
        print(f"{'dataset':32s} {'n_tr':>6s} {'n_te':>6s} {'cls':>5s} {'C':>6s} "
              f"{'acc':>7s} {'bal_acc':>8s}")
        for d in ds_dirs:
            r = probe_dataset(d, cs, args.max_iter, args.seed)
            if r is None:
                continue
            if "skipped" in r:
                print(f"{r['dataset'][:32]:32s} {'-- multi-label, needs mAP --':>36s}")
                continue
            r["run"] = run
            # strip the trailing _<model> so datasets line up across runs
            r["dataset"] = re.sub(r"_pupujepa.*$|_eat.*$|_beats.*$", "", r["dataset"])
            rows.append(r)
            print(f"{r['dataset'][:32]:32s} {r['n_train']:>6d} {r['n_test']:>6d} "
                  f"{r['n_classes']:>5d} {r['C']:>6g} {r['accuracy']:>7.3f} "
                  f"{r['balanced_accuracy']:>8.3f}")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "dataset", "n_train", "n_test",
                                              "n_classes", "C", "accuracy",
                                              "balanced_accuracy"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
