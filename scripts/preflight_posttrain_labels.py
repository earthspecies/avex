"""Preflight a post-training run's DATA before committing GPUs to it.

Builds the training data through exactly the code path `avex train` uses
(`RunConfig.from_sources` -> `build_dataloaders`), then checks the two things
that have silently broken post-training runs before:

1. LABELS. The joint label map must contain both AudioSet ontology classes and
   species, and sampled training rows must actually carry positive labels. The
   AAAI "All" data config built its map from `features: [labels]` only, which
   leaves every bioacoustic row with an
   EMPTY target -- every bioacoustic clip trained as an all-negative, with no
   error. (Species live in `canonical_name` for all four bioacoustic sources.) This catches that directly: a large empty-target fraction means the
   label transform is dropping a source.

2. DATA SIZE vs the paper. Reproducing AVEX means training on its mix (Table 4
   "Ours": AudioSet + Xeno-canto + Watkins + Animal Sound Archive + iNaturalist).
   Each source's size is printed next to paper Table 1 hours, so the known gap --
   today's Xeno-canto/iNaturalist are expanded post-paper dumps, not the June-2023
   snapshot of Appendix B.4 -- is measured rather than assumed.

Nothing here trains or writes; it only reads. Run it on the cluster (needs
alp-data and GCS access):

    uv run --group project-dev python scripts/preflight_posttrain_labels.py \
        --config configs/run_configs/icassp_2026/sl_eat_all_repro.yml

Exits non-zero if the label checks fail, so it can gate a launcher.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import torch

# AVEX paper Table 1, hours per post-training source (Table 4 "Ours": AS+XC+WTK+ASA+INAT).
PAPER_HOURS = {
    "audioset": 5700,
    "xeno-canto": 10416,
    "inaturalist": 1539,
    "watkins": 27,
    "animal-sound-archive": 78,
}

# Scientific binomial, e.g. "Turdus merula" / "Abroscopus superciliaris ss sakaiorum".
_BINOMIAL = re.compile(r"^[A-Z][a-z]+ [a-z]+(\b.*)?$")


def _positives(label: object) -> list[int]:
    """Return the positive class indices for one sample's label, whatever its encoding."""
    if isinstance(label, torch.Tensor):
        if label.dim() == 0:
            return [int(label)]
        if label.dtype.is_floating_point or label.numel() > 1:
            return torch.nonzero(label > 0, as_tuple=False).flatten().tolist()
        return label.flatten().tolist()
    if isinstance(label, (list, tuple)):
        return [int(x) for x in label]
    return [int(label)]


def _size_by_source(config: object) -> None:
    """Report each training source's size next to the paper's Table 1 hours.

    Builds every `train_datasets` entry standalone through alp-data (the same
    constructor training uses) and reports its row count, plus summed hours when
    the loaded table carries a duration-like column. alp-data declares no
    duration column for these datasets, so hours are best-effort; row counts are
    always shown. A source far larger than Table 1 is a newer, expanded dump.
    """
    from alp_data import DatasetConfig, dataset_from_config

    dc = config.dataset_config
    entries = getattr(dc, "train_datasets", None) or []
    print(f"  {'source':22s} {'rows':>11s} {'hours here':>11s} {'paper T1 h':>11s}")
    for entry in entries:
        name = entry.dataset_name
        try:
            fields = entry.model_dump(exclude={"transformations"})
            ds, _ = dataset_from_config(DatasetConfig(**fields))
            rows = len(ds)
            df = getattr(ds, "_data", None)
            cols = list(getattr(df, "columns", []))
            dur = next((c for c in cols if re.search(r"dur|length|seconds", c, re.I)), None)
            hours = "-"
            if dur is not None:
                total = float(df[dur].sum())
                # Heuristic: values look like milliseconds if the per-row mean is huge.
                scale = 3600_000.0 if total / max(rows, 1) > 1000 else 3600.0
                hours = f"{total / scale:,.0f}"
            print(f"  {name:22s} {rows:11,d} {hours:>11s} {PAPER_HOURS.get(name, '-')!s:>11s}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:22s} could not build standalone: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=400, help="random training rows to inspect")
    p.add_argument("--max-empty-frac", type=float, default=0.02, help="fail if more rows than this are empty")
    args = p.parse_args()

    from avex.configs import RunConfig
    from avex.data.dataset import build_dataloaders

    config = RunConfig.from_sources(yaml_file=args.config, cli_args=())
    train_dl, _, _ = build_dataloaders(config, device="cpu")
    ds = train_dl.dataset
    label_map: dict[str, int] = ds.metadata["label_map"]
    names = {v: k for k, v in label_map.items()}

    species = {i for i, n in names.items() if _BINOMIAL.match(n)}
    other = set(names) - species
    print(f"[label map] {len(label_map):,} classes: ~{len(species):,} species-like, ~{len(other):,} other (AudioSet)")
    print(f"  species sample : {[names[i] for i in sorted(species)[:4]]}")
    print(f"  AudioSet sample: {[names[i] for i in sorted(other)[:4]]}")

    ok = True
    if not species or not other:
        print("  FAIL: joint map is missing a whole label space -- a source's labels are being dropped")
        ok = False

    random.seed(0)
    idx = random.sample(range(len(ds)), min(args.samples, len(ds)))
    empty = only_sp = only_as = both = 0
    for i in idx:
        pos = set(_positives(ds[i]["label"]))
        if not pos:
            empty += 1
        elif pos <= species:
            only_sp += 1
        elif pos <= other:
            only_as += 1
        else:
            both += 1
    n = len(idx)
    print(f"[rows] sampled {n} of {len(ds):,} training rows")
    print(f"  empty target     : {empty:4d} ({empty / n:.1%})")
    print(f"  species-only     : {only_sp:4d} ({only_sp / n:.1%})")
    print(f"  AudioSet-only    : {only_as:4d} ({only_as / n:.1%})")
    print(f"  both spaces      : {both:4d} ({both / n:.1%})")
    if empty / n > args.max_empty_frac:
        print(f"  FAIL: {empty / n:.1%} of rows have no positive label (limit {args.max_empty_frac:.0%})")
        ok = False
    if only_sp == 0 or only_as == 0:
        print("  FAIL: sampled rows never carry labels from one of the two sources")
        ok = False

    print("[vintage]")
    _size_by_source(config)

    print("PREFLIGHT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
