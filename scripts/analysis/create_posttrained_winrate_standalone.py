#!/usr/bin/env python3
"""
create_posttrained_winrate_standalone.py
=======================================

Creates a standalone bar chart showing the benefit of post-training SSL backbones.
This is panel (a) from the post-trained win-rate analysis as a separate figure.

Usage:
    uv run python scripts/analysis/create_posttrained_winrate_standalone.py
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

# Configuration
EXCEL = Path("static/results/Representation Learning Results (35).xlsx")
OUT_BASE = Path("analysis")

# Post-training pairs: (post-trained model, base model)
POST_TRAINING_PAIRS = {
    "EAT-AS": ("sl-EAT-AS", "EAT-all"),  # Use EAT-all as base per user specification
    "EAT-bio": ("sl-EAT-bio", "EAT-all"),
    "EAT-all": ("sl-EAT-all", "EAT-all"),
    "BEATS-bio": ("sl-BEATS-bio", "BEATS (pretrained)"),
    "BEATS-all": ("sl-BEATS-all", "BEATS (pretrained)"),
}

# Benchmark-specific metrics (matching main analysis pipeline)
BENCHMARK_METRICS = {
    "BEANS Classification": ["Probe", "R-auc", "C-nmi"],
    "BEANS Detection": ["Probe", "R-auc"],  # No C-nmi for detection
    "BirdSet": ["Probe", "R-auc"],  # No C-nmi for detection  
    "Individual ID": ["Probe", "R-auc"],
    "Vocal Repertoire": ["R-auc", "C-nmi"]  # No Probe for vocal repertoire
}

AGGREGATE_PREFIXES = {
    "BEANS Classification",
    "BEANS Detection", 
    "BirdSet",
    "Individual ID",
    "Repertoire",
    "Vocal Repertoire",
}

# Benchmark groups for win-rate analysis
BENCHMARK_GROUPS = {
    "BEANS Classification": ["Watkins", "CBI", "HBDB", "BATS", "Dogs", "ESC-50"],
    "BEANS Detection": ["enabirds", "rfcx", "hiceas", "gibbons", "dcase"],
    "BirdSet": ["POW", "PER", "NES", "NBP", "HSN", "SNE", "UHH"],
    "Individual ID": ["chiffchaff-cross", "littleowls-cross", "pipit-cross", "macaques"],
    "Vocal Repertoire": ["zebrafinch-je-call", "Giant_Otters", "Bengalese_Finch", "SRKW_Orca"]
}


def load_df(path: Path) -> pd.DataFrame:
    """Flatten Excel into model × metric DataFrame."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    headers: list[str | None] = []
    for col in range(3, ws.max_column + 1):
        ds = ws.cell(row=3, column=col).value
        met = ws.cell(row=4, column=col).value
        headers.append(f"{str(ds).replace(' ', '_')}_{str(met).replace(' ', '_')}" if ds and met else None)
    rows = []
    for row in range(5, ws.max_row + 1):
        model = ws.cell(row=row, column=2).value
        if not model:
            continue
        rec: dict[str, float] = {"model": model}
        for idx, hdr in enumerate(headers, start=3):
            if hdr is None:
                continue
            val = ws.cell(row=row, column=idx).value
            try:
                rec[hdr] = float(val) if val not in (None, "", "N/A") else np.nan
            except Exception:
                rec[hdr] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("model")


def compute_benchmark_win_rates(df: pd.DataFrame, post_trained: str, base: str) -> Dict[str, Tuple[int, int, float]]:
    """Compute win-rates by benchmark group using appropriate metrics for each benchmark."""
    if post_trained not in df.index or base not in df.index:
        return {}
    
    g = pd.to_numeric(df.loc[post_trained], errors="coerce")
    b = pd.to_numeric(df.loc[base], errors="coerce")
    imp = (g - b) / b.replace(0, np.nan) * 100
    
    benchmark_wins = {}
    
    for benchmark_name, datasets in BENCHMARK_GROUPS.items():
        # Get metrics for this specific benchmark
        valid_metrics = BENCHMARK_METRICS.get(benchmark_name, [])
        
        # Collect improvement values for this benchmark
        benchmark_improvements = []
        
        for metric in valid_metrics:
            if benchmark_name == "Individual ID" and metric == "R-auc":
                # Special handling: prefer R-cross-auc over R-auc when available (matching main analysis)
                for dataset in datasets:
                    cross_key = f"{dataset}_R-cross-auc"
                    regular_key = f"{dataset}_R-auc"
                    
                    if cross_key in imp.index and pd.notna(imp[cross_key]):
                        # Use cross-auc when available (more robust)
                        benchmark_improvements.append(imp[cross_key])
                    elif regular_key in imp.index and pd.notna(imp[regular_key]):
                        # Fall back to regular R-auc
                        benchmark_improvements.append(imp[regular_key])
            else:
                # Standard handling for all other benchmark/metric combinations
                for dataset in datasets:
                    col_name = f"{dataset}_{metric}"
                    if col_name in imp.index and pd.notna(imp[col_name]):
                        benchmark_improvements.append(imp[col_name])
        
        if benchmark_improvements:
            # Convert to series for win_rate calculation
            benchmark_series = pd.Series(benchmark_improvements)
            wins = int((benchmark_series > 0).sum())
            total = int(benchmark_series.size)
            avg_imp = benchmark_series.mean()
            benchmark_wins[benchmark_name] = (wins, total, avg_imp)
    
    return benchmark_wins


def aggregate_win_rates_across_models(df: pd.DataFrame) -> Dict[str, Tuple[int, int, float]]:
    """Compute aggregated win-rates across all post-training pairs for each benchmark."""
    
    aggregated_wins = {}
    
    # Initialize counters for each benchmark
    for benchmark_name in BENCHMARK_GROUPS.keys():
        total_wins = 0
        total_comparisons = 0
        all_improvements = []
        
        # Compute win-rates for each post-training pair
        for model_name, (post_trained, base) in POST_TRAINING_PAIRS.items():
            benchmark_wins = compute_benchmark_win_rates(df, post_trained, base)
            
            if benchmark_name in benchmark_wins:
                wins, total, avg_imp = benchmark_wins[benchmark_name]
                total_wins += wins
                total_comparisons += total
                
                # Collect individual improvements for overall average
                g = pd.to_numeric(df.loc[post_trained], errors="coerce")
                b = pd.to_numeric(df.loc[base], errors="coerce")
                imp = (g - b) / b.replace(0, np.nan) * 100
                
                # Get improvements for this benchmark (same logic as compute_benchmark_win_rates)
                valid_metrics = BENCHMARK_METRICS.get(benchmark_name, [])
                for metric in valid_metrics:
                    if benchmark_name == "Individual ID" and metric == "R-auc":
                        for dataset in BENCHMARK_GROUPS[benchmark_name]:
                            cross_key = f"{dataset}_R-cross-auc"
                            regular_key = f"{dataset}_R-auc"
                            
                            if cross_key in imp.index and pd.notna(imp[cross_key]):
                                all_improvements.append(imp[cross_key])
                            elif regular_key in imp.index and pd.notna(imp[regular_key]):
                                all_improvements.append(imp[regular_key])
                    else:
                        for dataset in BENCHMARK_GROUPS[benchmark_name]:
                            col_name = f"{dataset}_{metric}"
                            if col_name in imp.index and pd.notna(imp[col_name]):
                                all_improvements.append(imp[col_name])
        
        if total_comparisons > 0:
            avg_improvement = np.mean(all_improvements) if all_improvements else 0.0
            aggregated_wins[benchmark_name] = (total_wins, total_comparisons, avg_improvement)
    
    return aggregated_wins


def create_standalone_winrate_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create standalone bar chart showing benefit of post-training SSL backbones."""
    
    # Compute aggregated win-rates across all post-training pairs
    aggregated_wins = aggregate_win_rates_across_models(df)
    
    if not aggregated_wins:
        print("No benchmark win-rate data available!")
        return
    
    # === CREATE PLOT ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Extract data for plotting
    groups = list(aggregated_wins.keys())
    win_rates = [wins/total*100 for wins, total, _ in aggregated_wins.values()]
    avg_gains = [avg_gain for _, _, avg_gain in aggregated_wins.values()]
    counts = [f"{wins}/{total}" for wins, total, _ in aggregated_wins.values()]
    
    # Shorten group names for better display
    display_groups = []
    for group in groups:
        if group == "BEANS Classification":
            display_groups.append("BEANS\nClassification")
        elif group == "BEANS Detection":
            display_groups.append("BEANS\nDetection")
        elif group == "Individual ID":
            display_groups.append("Individual\nID")
        elif group == "Vocal Repertoire":
            display_groups.append("Vocal\nRepertoire")
        else:
            display_groups.append(group)
    
    # Use ESP 5-tone palette for bars: Blue 4, Cyan 2, Blue 1, Cyan 1, Blue 2
    bars = ax.bar(display_groups, win_rates, 
                  color=['#00738B', '#04CDA0', '#C6DEE7', '#1ADCCF', '#98C6D2'], 
                  alpha=1.0, edgecolor='black', linewidth=1)
    
    # Add labels on bars
    for bar, count, avg_gain in zip(bars, counts, avg_gains):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}%\n{count}\n{avg_gain:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Win-Rate (%)', fontweight='bold', fontsize=14)
    ax.set_title('Benefit of Post-training SSL Backbones', fontweight='bold', fontsize=16, pad=20)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    filename = "posttrained_winrate_standalone.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created standalone post-training win-rate plot: {filename}")


def main() -> None:
    """Main execution function."""
    print("🔬 Creating Standalone Post-training Win-Rate Analysis...")
    
    # Load data
    df = load_df(EXCEL)
    
    # Create output directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"{ts}_posttrained_winrate_standalone"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the standalone plot
    create_standalone_winrate_plot(df, out_dir)
    
    print(f"✅ Analysis complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()





