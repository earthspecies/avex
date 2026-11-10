"""Linear-probe cached, mean-pooled avex embeddings -- raw vs z-scored features.

Why this exists: on EAT's blocks.11.attn.proj the pooled features are tiny
(L2 norm ~1 for EAT-all SSL), and the paper-spec probe (AdamW lr 1e-4, wd 0.1,
bs 32, 900 epochs) UNDERFITS them -- e.g. EAT-all SSL on Dogs reaches only 49%
train accuracy. Z-scoring each feature with train-split statistics fixes it
(Dogs 45 -> 80 test, Watkins 65 -> 91), landing at the paper's numbers. This
script re-runs both variants from the h5 caches written by `avex evaluate`
(offline_embeddings.probe_storage_aggregation: mean), so nothing is re-extracted.

The raw variant reproduces avex's own probe (Watkins 65.19 in both). Selection
is by best val accuracy, as in avex.

Usage:
    uv run --group project-dev python scripts/probe_cached_embeddings.py \
        --cache-dir evaluation_results/icassp2026_eat_ssl_all_probe_attnproj_pooled \
        --layer backbone.model.blocks.11.attn.proj --seeds 0 1 2 \
        --out evaluation_results/icassp2026_eat_ssl_all_probe_attnproj_pooled/zscore_probe.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import torch

from avex.metrics.beans_metrics import MeanAveragePrecision


def load_split(ds_dir: Path, split: str, layer: str, multi_label: bool) -> tuple[torch.Tensor, torch.Tensor]:
    layer_id = layer.replace(".", "_").replace("backbone_", "")
    with h5py.File(ds_dir / f"embedding_{split}_{layer_id}.h5") as f:
        x = torch.tensor(f[f"embeddings_{layer}"][:], dtype=torch.float32)
        y = torch.tensor(f["labels"][:])
    if x.dim() != 2:
        raise ValueError(f"{ds_dir.name}/{split}: expected pooled (N, D) embeddings, got {tuple(x.shape)}")
    if multi_label:  # detection: keep the binary (N, C) target matrix
        if y.dim() != 2 or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError(f"{ds_dir.name}/{split}: expected binary (N, C) multi-label targets")
        return x, y.float()
    if y.dim() == 2:  # one-hot -> index (BEANS classification is single-label)
        if not bool((y.sum(1) == 1).all()):
            raise ValueError(f"{ds_dir.name}/{split}: labels are not single-label one-hot")
        y = y.argmax(1)
    return x, y


def _map(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Macro mAP exactly as avex scores detection probes (sigmoid + per-class AP)."""
    m = MeanAveragePrecision()
    m.update(logits.detach().float().cpu(), y.long().cpu())
    return float(m.get_primary_metric())


def run_probe(xtr, ytr, xva, yva, xte, yte, *, lr, wd, epochs, bs, seed, multi_label) -> dict[str, float]:
    """Paper-spec linear probe. Single-label: cross-entropy, scored by accuracy.
    Multi-label (detection): BCE-with-logits, scored by macro mAP -- the loss and
    metric avex's own probe uses for these datasets. Best epoch by the val score."""
    torch.manual_seed(seed)
    n_cls = ytr.shape[1] if multi_label else int(max(ytr.max(), yva.max(), yte.max())) + 1
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits if multi_label else torch.nn.functional.cross_entropy
    score = _map if multi_label else (lambda logits, y: (logits.argmax(1) == y).float().mean().item())
    model = torch.nn.Linear(xtr.shape[1], n_cls).to(xtr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(xtr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6)
    best_val, best_state = -1.0, None
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=xtr.device)
        for i in range(0, len(xtr), bs):
            b = perm[i : i + bs]
            loss = loss_fn(model(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            val = score(model(xva), yva)
        if val > best_val:
            best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        return {"metric": "mAP" if multi_label else "accuracy", "train_score": score(model(xtr), ytr),
                "val_score": best_val, "test_score": score(model(xte), yte)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=900)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu", help="cpu or cuda; all tensors live on it")
    p.add_argument("--variants", nargs="+", default=["raw", "zscore"], choices=["raw", "zscore"])
    p.add_argument("--run-name", default=None,
                   help="probe EVERY cached dataset dir '<dataset>_<run-name>' (single-label unless it has "
                        "'_detection'), instead of only *_classification_* / *_detection_* dirs -- needed "
                        "for Table 9 (pipit_id_across_year, macaques_individual_id, ...)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if args.run_name:
        suffix = "_" + args.run_name
        # Only dirs with a probe cache for this layer (retrieval/clustering-only sets such as
        # Bengalese Finch have just embedding_*_clustering.h5).
        layer_id = args.layer.replace(".", "_").replace("backbone_", "")
        ds_dirs = sorted(d for d in args.cache_dir.iterdir() if d.is_dir() and d.name.endswith(suffix)
                         and (d / f"embedding_train_{layer_id}.h5").exists())
    else:
        ds_dirs = sorted(d for d in args.cache_dir.iterdir()
                         if d.is_dir() and ("_classification_" in d.name or "_detection_" in d.name))
    if not ds_dirs:
        raise SystemExit(f"no *_classification_* / *_detection_* cache dirs in {args.cache_dir}")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    rows = []
    for d in ds_dirs:
        kind = "_detection" if "_detection_" in d.name else "_classification"
        multi = kind == "_detection"
        if args.run_name:
            dataset = d.name[: -len(args.run_name) - 1]
        (xtr, ytr), (xva, yva), (xte, yte) = (
            tuple(t.to(args.device) for t in load_split(d, s, args.layer, multi)) for s in ("train", "val", "test")
        )
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        variants = {"raw": (xtr, xva, xte), "zscore": ((xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd)}
        variants = {k: v for k, v in variants.items() if k in args.variants}
        if not args.run_name:
            dataset = d.name.split(kind + "_")[0] + kind
        for variant, (a, b, c) in variants.items():
            for seed in args.seeds:
                r = run_probe(a, ytr, b, yva, c, yte, lr=args.lr, wd=args.wd, epochs=args.epochs, bs=args.bs,
                              seed=seed, multi_label=multi)
                rows.append({"dataset": dataset, "variant": variant, "seed": seed, "n_train": len(xtr),
                             "feat_norm": round(xtr.norm(dim=1).mean().item(), 4), **r})
                print(f"{dataset:30s} {variant:6s} seed {seed} [{r['metric']}]: train {100 * r['train_score']:6.2f}  "
                      f"val {100 * r['val_score']:6.2f}  test {100 * r['test_score']:6.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
