"""Print AVEX Table 6 rows (BEANS classification: Probe / R-AUC / NMI per dataset)
for our EAT evaluations, next to the paper's rows.

Probe  = linear-probe test accuracy on z-scored features, mean over seeds
         (scripts/probe_cached_embeddings.py -> zscore_probe.csv)
R-AUC  = retrieval ROC-AUC, NMI = k-means clustering NMI
         (avex evaluate, layer blocks.11.attn.proj -> results.csv)

Prints a percentage table for reading and LaTeX rows in the paper's 0.xxx format
for pasting into Table 6.
"""

from pathlib import Path

import pandas as pd

R = Path("evaluation_results")
DATASETS = [  # Table 6 column order
    ("marine_mammal_classification", "Watkins"),
    ("bird_classification", "CBI"),
    ("mosquito_classification", "HBDB"),
    ("bat_classification", "BATS"),
    ("dog_classification", "Dogs"),
    ("esc50_classification", "ESC-50"),
]
# Paper Table 6, (Probe, R-AUC, NMI) per dataset in the order above.
PAPER = {
    "EAT-all (paper)": [(.873, .773, .618), (.326, .644, .516), (.791, .722, .475),
                        (.655, .612, .162), (.755, .593, .227), (.853, .878, .689)],
    "sl-EAT-all (paper)": [(.885, .932, .761), (.755, .943, .802), (.754, .657, .340),
                           (.650, .635, .183), (.863, .681, .384), (.818, .895, .747)],
}
OURS = [  # (label, eval tag)
    ("EAT-all (ours)", "eat_ssl_all"),
    ("sl-EAT-all published ckpt (ours)", "eat_published_sl_all"),
    ("sl-EAT-all our post-train (ours)", "eat_repro_sl_all"),
]


def ours(tag: str) -> list[tuple[float, float, float]]:
    z = pd.read_csv(R / f"icassp2026_{tag}_probe_attnproj_pooled/zscore_probe.csv")
    probe = z[z.variant == "zscore"].groupby("dataset")["test_score" if "test_score" in z else "test_acc"].mean()
    r = pd.read_csv(R / f"icassp2026_{tag}_retrieval_attnproj/results.csv")
    r = r.drop_duplicates("evaluation_dataset_name", keep="last").set_index("evaluation_dataset_name")
    return [(probe[k], r.loc[k, "retrieval_roc_auc"], r.loc[k, "clustering_nmi"]) for k, _ in DATASETS]


rows = {}
for label, tag in OURS:
    try:
        rows[label] = ours(tag)
    except (FileNotFoundError, KeyError) as e:
        print(f"[skip] {label}: {type(e).__name__} {e}")
order = ["EAT-all (paper)", "EAT-all (ours)", "sl-EAT-all (paper)",
         "sl-EAT-all published ckpt (ours)", "sl-EAT-all our post-train (ours)"]
rows = {k: (PAPER.get(k) or rows.get(k)) for k in order if (k in PAPER or k in rows)}

hdr = " | ".join(f"{n} Probe | {n} R-AUC | {n} NMI" for _, n in DATASETS)
print(f"| Model | {hdr} |\n|---|" + "---|" * 18)
for k, v in rows.items():
    print(f"| {k} | " + " | ".join(f"{100 * x:.2f}" for t in v for x in t) + " |")

print("\n% LaTeX rows for Table 6 (paper format)")
for k, v in rows.items():
    if "(ours)" in k:
        print(f"{k} & " + " & ".join(f"{x:.3f}" for t in v for x in t) + r" \\")
