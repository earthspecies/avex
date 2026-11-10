"""Print AVEX Table 6 rows (BEANS classification: Probe / R-AUC / NMI per dataset).

Each --row is LABEL=DIR or LABEL=PROBE_DIR,RETRIEVAL_DIR:
  PROBE_DIR/zscore_probe.csv  -> Probe (z-scored linear probe, mean over seeds)
  RETRIEVAL_DIR/results.csv   -> R-AUC (retrieval_roc_auc), NMI (clustering_nmi)
A missing file prints "--" for those cells instead of failing, so the table can
be run while jobs are still going.

Example (the overnight PupuJEPA jobs write probe + retrieval into one dir):
    uv run python scripts/compile_table6.py \
        --row "PupuJEPA SSL, final=evaluation_results/icassp2026_pupujepa_bigrun_final" \
        --row "PupuJEPA SSL, attn.proj=evaluation_results/icassp2026_pupujepa_bigrun_attnproj" \
        --row "sl-PupuJEPA-all, final=evaluation_results/icassp2026_pupujepa_avexall_final" \
        --row "sl-PupuJEPA-all, attn.proj=evaluation_results/icassp2026_pupujepa_avexall_attnproj"
"""

import argparse
from pathlib import Path

import pandas as pd

TABLES = {  # paper column order; Table 6 has NMI, Tables 7-8 (detection, mAP probe) do not
    "6": ([("marine_mammal_classification", "Watkins"), ("bird_classification", "CBI"),
           ("mosquito_classification", "HBDB"), ("bat_classification", "BATS"),
           ("dog_classification", "Dogs"), ("esc50_classification", "ESC-50")], True),
    "7": ([("enabirds_detection", "enabirds"), ("rfcx_detection", "rfcx"), ("hiceas_detection", "hiceas"),
           ("hainan_gibbons_detection", "gibbons"), ("dcase_detection", "dcase")], False),
    "8": ([(f"birdset_{s.lower()}_detection", s) for s in ("POW", "PER", "NES", "NBP", "HSN", "SNE", "UHH")], False),
}
DATASETS, WITH_NMI = TABLES["6"]

# Reference rows copied from the AVEX paper (Tables 6-8), in the column order above.
PAPER = {
    "6": {"EAT-all (paper)": [.873, .773, .618, .326, .644, .516, .791, .722, .475, .655, .612, .162, .755, .593, .227, .853, .878, .689],
          "sl-EAT-all (paper)": [.885, .932, .761, .755, .943, .802, .754, .657, .340, .650, .635, .183, .863, .681, .384, .818, .895, .747]},
    "7": {"EAT-all (paper)": [.475, .668, .103, .723, .569, .511, .155, .666, .275, .901],
          "sl-EAT-all (paper)": [.528, .665, .099, .739, .536, .618, .170, .667, .445, .832]},
    "8": {"EAT-all (paper)": [.188, .702, .065, .649, .113, .731, .303, .648, .185, .568, .147, .708, .158, .734],
          "sl-EAT-all (paper)": [.265, .700, .129, .600, .219, .828, .452, .707, .328, .586, .192, .760, .203, .763]},
}
# Table 9 mixes metrics and sources, so each column names its dataset and metric explicitly:
#   ("probe", ds)  z-scored probe accuracy;  ("rauc", ds) retrieval ROC-AUC;  ("nmi", ds) clustering NMI.
# R-AUC* (the -cross ID sets) is retrieval with TRAIN items as queries = the *_train_vs_test sets.
T9_COLS = [
    ("chiffchaff-cross Probe", "probe", "chiffchaff_id_across_year"),
    ("chiffchaff-cross R-AUC*", "rauc", "chiffchaff_id_across_year_train_vs_test"),
    ("littleowls-cross Probe", "probe", "littleowl_id_across_year"),
    ("littleowls-cross R-AUC*", "rauc", "littleowl_id_across_year_train_vs_test"),
    ("pipit-cross Probe", "probe", "pipit_id_across_year"),
    ("pipit-cross R-AUC*", "rauc", "pipit_id_across_year_train_vs_test"),
    ("macaques Probe", "probe", "macaques_individual_id"),
    ("macaques R-AUC", "rauc", "macaques_individual_id"),
    ("zebrafinch-je-call R-AUC", "rauc", "zebra_finch_call_type"),
    ("zebrafinch-je-call NMI", "nmi", "zebra_finch_call_type"),
    ("Giant Otters R-AUC", "rauc", "giant_otters_vocalization"),
    ("Giant Otters NMI", "nmi", "giant_otters_vocalization"),
    ("Bengalese Finch R-AUC", "rauc", "bengalese_finch_repertoire_{bird}"),
    ("Bengalese Finch NMI", "nmi", "bengalese_finch_repertoire_{bird}"),
    ("SRKW Orca R-AUC", "rauc", "vfpa_srkw_call_types"),   # not evaluated: dataset unavailable
    ("SRKW Orca NMI", "nmi", "vfpa_srkw_call_types"),
]
PAPER["9"] = {"EAT-all (paper)": [.200, .547, .152, .575, .109, .487, .929, .836, .709, .333, .820, .549, .977, .847, .646, .321],
              "sl-EAT-all (paper)": [.195, .509, .354, .688, .326, .557, .949, .795, .718, .338, .789, .501, .980, .898, .703, .383]}


def row9(d: Path, bird: str) -> list[float | None]:
    """One Table 9 row from a combined eval dir (zscore_probe.csv + results.csv)."""
    probe, rauc, nmi = {}, {}, {}
    if (d / "zscore_probe.csv").exists():
        z = pd.read_csv(d / "zscore_probe.csv")
        probe = z[z.variant == "zscore"].groupby("dataset")["test_score" if "test_score" in z else "test_acc"].mean().to_dict()
    if (d / "results.csv").exists():
        r = pd.read_csv(d / "results.csv").drop_duplicates("evaluation_dataset_name", keep="last")
        rauc = dict(zip(r.evaluation_dataset_name, r.retrieval_roc_auc))
        if "clustering_nmi" in r:
            nmi = dict(zip(r.evaluation_dataset_name, r.clustering_nmi))
    src = {"probe": probe, "rauc": rauc, "nmi": nmi}
    out = [src[m].get(ds.format(bird=bird)) for _, m, ds in T9_COLS]
    return [None if (v is None or pd.isna(v)) else float(v) for v in out]


R = "evaluation_results/icassp2026_"
# Our runs, used when no --row is given. Table 6 probe and retrieval live in separate dirs for EAT.
DEFAULT_ROWS = {
    "6": [("EAT-all (ours)", f"{R}eat_ssl_all_probe_attnproj_pooled,{R}eat_ssl_all_retrieval_attnproj"),
          ("sl-EAT-all (ours)", f"{R}eat_repro_sl_all_probe_attnproj_pooled,{R}eat_repro_sl_all_retrieval_attnproj"),
          ("PupuJEPA SSL (ours)", f"{R}pupujepa_bigrun_final"),
          ("sl-PupuJEPA-all (ours)", f"{R}pupujepa_avexall_final"),
          ("XC_1M MAE SSL (Moritz)", f"{R}xenomae_xc1m_t6"),
          ("sl-XC_1M MAE-all (ours)", f"{R}xenomae_avexall_t6"),
          ("EffNetB0-32k (Marius)", f"{R}effnet32k_ssl_t6"),
          ("sl-EffNetB0-32k-all (ours)", f"{R}effnet32k_avexall_t6")],
    **{tb: [("EAT-all (ours)", f"{R}eat_ssl_all_t{tb}"), ("sl-EAT-all (ours)", f"{R}eat_repro_sl_all_t{tb}"),
            ("PupuJEPA SSL (ours)", f"{R}pupujepa_bigrun_t{tb}"), ("sl-PupuJEPA-all (ours)", f"{R}pupujepa_avexall_t{tb}"),
            ("XC_1M MAE SSL (Moritz)", f"{R}xenomae_xc1m_t{tb}"), ("sl-XC_1M MAE-all (ours)", f"{R}xenomae_avexall_t{tb}"),
            ("EffNetB0-32k (Marius)", f"{R}effnet32k_ssl_t{tb}"), ("sl-EffNetB0-32k-all (ours)", f"{R}effnet32k_avexall_t{tb}")]
       for tb in ("7", "8", "9")},
}
# Print order: each paper row followed by our run of the same model.
ORDER = ["EAT-all (paper)", "EAT-all (ours)", "sl-EAT-all (paper)", "sl-EAT-all (ours)",
         "PupuJEPA SSL (ours)", "sl-PupuJEPA-all (ours)", "XC_1M MAE SSL (Moritz)", "sl-XC_1M MAE-all (ours)",
         "EffNetB0-32k (Marius)", "sl-EffNetB0-32k-all (ours)"]


def row(probe_dir: Path, retr_dir: Path) -> list[float | None]:
    probe, ret = {}, {}
    zf = probe_dir / "zscore_probe.csv"
    if zf.exists():
        z = pd.read_csv(zf)
        probe = z[z.variant == "zscore"].groupby("dataset")["test_score" if "test_score" in z else "test_acc"].mean().to_dict()
    rf = retr_dir / "results.csv"
    if rf.exists():
        r = pd.read_csv(rf).drop_duplicates("evaluation_dataset_name", keep="last")
        nmi = r["clustering_nmi"] if "clustering_nmi" in r else [None] * len(r)  # absent without clustering
        ret = {k: (a, n) for k, a, n in zip(r.evaluation_dataset_name, r.retrieval_roc_auc, nmi)}
    out: list[float | None] = []
    for k, _ in DATASETS:
        a, n = ret.get(k, (None, None))
        out += [probe.get(k), a, n] if WITH_NMI else [probe.get(k), a]
    return [None if (v is None or pd.isna(v)) else float(v) for v in out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", help="LABEL=DIR or LABEL=PROBE_DIR,RETRIEVAL_DIR "
                   "(default: our four models + the paper's EAT rows)")
    p.add_argument("--table", choices=sorted(TABLES) + ["9"], default="6",
                   help="paper table: 6 BEANS cls, 7 BEANS det, 8 BirdSet, 9 Individual ID + Vocal Repertoire")
    p.add_argument("--bengalese", choices=["bird0", "bird2"], default="bird0",
                   help="Table 9: which Bengalese Finch bird fills the LaTeX column (both are printed)")
    args = p.parse_args()
    if args.table == "9":
        return main9(args)
    global DATASETS, WITH_NMI
    DATASETS, WITH_NMI = TABLES[args.table]
    k = 3 if WITH_NMI else 2
    rows = {}
    specs = args.row or [f"{lab}={d}" for lab, d in DEFAULT_ROWS[args.table]]
    for spec in specs:
        label, dirs = spec.split("=", 1)
        parts = dirs.split(",")
        rows[label] = row(Path(parts[0]), Path(parts[-1]))
    if not args.row:
        rows.update({k: list(v) for k, v in PAPER[args.table].items()})
        rows = {k: rows[k] for k in ORDER if k in rows}

    cols = ["Probe", "R-AUC", "NMI"][:k]
    hdr = " | ".join(f"{n} {c}" for _, n in DATASETS for c in cols)
    print(f"| Model | {hdr} | " + " | ".join(f"mean {c}" for c in cols) + " |\n|---|" + "---|" * (len(DATASETS) * k + k))
    for label, v in rows.items():
        cells = " | ".join("--" if x is None else f"{100 * x:.2f}" for x in v)
        means = []
        for i in range(k):
            col = [x for x in v[i::k] if x is not None]
            means.append(f"{100 * sum(col) / len(col):.2f}" if len(col) == len(DATASETS) else "--")
        print(f"| {label} | {cells} | " + " | ".join(means) + " |")
    print(f"\n% LaTeX rows for Table {args.table} (paper format)")
    for label, v in rows.items():
        if "(paper)" in label:
            continue
        print(f"{label} & " + " & ".join("--" if x is None else f"{x:.3f}" for x in v) + r" \\")


def main9(args: argparse.Namespace) -> None:
    rows: dict[str, list] = {}
    for lab, d in DEFAULT_ROWS["9"]:
        rows[lab] = row9(Path(d), args.bengalese)
        rows[lab + " [Bengalese " + ("bird0" if args.bengalese == "bird2" else "bird2") + "]"] = None
    other = "bird0" if args.bengalese == "bird2" else "bird2"
    rows.update({k: list(v) for k, v in PAPER["9"].items()})
    order = [k for k in ORDER if k in rows]
    hdr = [c for c, _, _ in T9_COLS] + [f"Bengalese ({other}) R-AUC", f"Bengalese ({other}) NMI"]
    print("| Model | " + " | ".join(hdr) + " |\n|---|" + "---|" * len(hdr))
    for lab in order:
        v = rows[lab]
        alt = row9(Path(dict(DEFAULT_ROWS["9"])[lab]), other)[12:14] if "(paper)" not in lab else [None, None]
        print(f"| {lab} | " + " | ".join("--" if x is None else f"{100 * x:.2f}" for x in v + alt) + " |")
    print(f"\n% LaTeX rows for Table 9 (paper format; Bengalese = {args.bengalese}; SRKW Orca not evaluated)")
    for lab in order:
        if "(paper)" in lab:
            continue
        print(f"{lab} & " + " & ".join("--" if x is None else f"{x:.3f}" for x in rows[lab]) + r" \\")


if __name__ == "__main__":
    main()
