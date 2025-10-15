#!/usr/bin/env python3
"""
Visualize detector-classifier predictions vs ground truth.

This script creates a visualization with:
1. Spectrogram
2. Top 5 species probability lines
3. Ground truth events (colored by species)
4. Predicted events (colored by species)
5. Animal detection probability heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import librosa
import librosa.display
from typing import List, Dict, Tuple


def get_top_species(species_probs: np.ndarray, output_species_labels: List[str], top_k: int = 5) -> List[int]:
    """
    Get indices of top K species by maximum probability across all frames.
    
    Parameters
    ----------
    species_probs : np.ndarray
        Shape (num_frames, num_classes)
    output_species_labels : List[str]
        Species names
    top_k : int
        Number of top species to return
        
    Returns
    -------
    List[int]
        Indices of top K species
    """
    max_probs = species_probs.max(axis=0)
    top_indices = np.argsort(max_probs)[-top_k:][::-1]
    return top_indices.tolist()


def threshold_and_merge_events(
    final_scores: np.ndarray,
    output_species_labels: List[str],
    frame_rate: float,
    threshold: float
) -> List[Tuple[float, float, str]]:
    """
    Threshold final scores and merge consecutive frames into events.
    
    Parameters
    ----------
    final_scores : np.ndarray
        Shape (num_frames, num_classes)
    output_species_labels : List[str]
        Species names corresponding to columns
    frame_rate : float
        Frames per second
    threshold : float
        Detection threshold
        
    Returns
    -------
    List[Tuple[float, float, str]]
        List of (start_time, end_time, species_name) tuples
    """
    num_frames, num_classes = final_scores.shape
    events = []
    
    # For each species, find events
    for species_idx, species_name in enumerate(output_species_labels):
        species_scores = final_scores[:, species_idx]
        above_threshold = species_scores >= threshold
        
        if not above_threshold.any():
            continue
        
        # Find runs of consecutive True values
        # Pad with False to handle edges
        padded = np.concatenate([[False], above_threshold, [False]])
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start_frame, end_frame in zip(starts, ends):
            start_time = start_frame / frame_rate
            end_time = end_frame / frame_rate
            
            # For this event, find the species with highest average probability
            event_scores = final_scores[start_frame:end_frame, :]
            avg_scores = event_scores.mean(axis=0)
            best_species_idx = avg_scores.argmax()
            best_species = output_species_labels[best_species_idx]
            
            events.append((start_time, end_time, best_species))
    
    # Merge overlapping events with same species
    if not events:
        return []
    
    events.sort()
    merged = []
    current_start, current_end, current_species = events[0]
    
    for start, end, species in events[1:]:
        # If same species and overlapping/adjacent, merge
        if species == current_species and start <= current_end + 0.01:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end, current_species))
            current_start, current_end, current_species = start, end, species
    
    merged.append((current_start, current_end, current_species))
    
    return merged


def get_species_colors(all_species: List[str]) -> Dict[str, tuple]:
    """
    Assign colors to species using a colormap.
    
    Parameters
    ----------
    all_species : List[str]
        List of unique species names
        
    Returns
    -------
    Dict[str, tuple]
        Mapping from species name to RGBA color
    """
    if not all_species:
        return {}
    
    # Use tab20 colormap for good distinction between species
    cmap = cm.get_cmap('tab20' if len(all_species) <= 20 else 'hsv')
    colors = {}
    
    for i, species in enumerate(sorted(all_species)):
        colors[species] = cmap(i / max(len(all_species), 1))
    
    return colors


def plot_visualization(
    audio: np.ndarray,
    sr: int,
    species_probs: np.ndarray,
    output_species_labels: List[str],
    animal_probs: np.ndarray,
    selection_table: pd.DataFrame,
    annotation_column: str,
    final_scores: np.ndarray,
    frame_rate: float,
    threshold: float,
    output_path: str
) -> None:
    """
    Create visualization with spectrogram, species probabilities, and events.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    species_probs : np.ndarray
        Shape (num_frames, num_classes), per-frame species probabilities
    output_species_labels : List[str]
        Species names corresponding to columns in species_probs
    animal_probs : np.ndarray
        Shape (num_frames,), per-frame animal detection probabilities
    selection_table : pd.DataFrame
        Ground truth with columns: "Begin Time (s)", "End Time (s)", annotation_column
    annotation_column : str
        Column name for species labels in selection_table
    final_scores : np.ndarray
        Shape (num_frames, num_classes), combined detection × classification scores
    frame_rate : float
        Frames per second
    threshold : float
        Detection threshold
    output_path : str
        Path to save the plot
    """
    duration = len(audio) / sr
    num_frames = len(animal_probs)
    
    # Create time array for frame-level data
    frame_times = np.arange(num_frames) / frame_rate
    
    # Get top 5 species
    top_species_indices = get_top_species(species_probs, output_species_labels, top_k=5)
    top_5_species = set([output_species_labels[idx] for idx in top_species_indices])
    
    # Get predicted events
    pred_events = threshold_and_merge_events(
        final_scores, output_species_labels, frame_rate, threshold
    )
    
    # Get all unique species from top 5, ground truth, and predictions
    gt_species = set(selection_table[annotation_column].unique())
    pred_species = set([species for _, _, species in pred_events])
    all_species = sorted(gt_species | pred_species | top_5_species)
    
    # Assign colors to all species that appear anywhere
    species_colors = get_species_colors(all_species)
    
    # Create figure with 4 rows
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(4, 1, height_ratios=[4, 2, 0.5, 0.5], hspace=0.35)
    
    # ============= ROW 1: SPECTROGRAM =============
    ax_spec = fig.add_subplot(gs[0])
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                            ax=ax_spec, cmap='viridis')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(f'Detector-Classifier Visualization (Threshold={threshold:.2f})')
    ax_spec.set_xlabel('')
    
    # ============= ROW 2: TOP 5 SPECIES PROBABILITIES + ANIMAL DETECTION =============
    ax_species = fig.add_subplot(gs[1], sharex=ax_spec)
    
    # Plot animal detection probability as a bold black line
    ax_species.plot(frame_times, animal_probs, label='Animal Detection', 
                   color='black', linewidth=4, alpha=0.9, linestyle='-', zorder=10)
    
    # Plot top 5 species probabilities
    for idx in top_species_indices:
        species_name = output_species_labels[idx]
        probs = species_probs[:, idx]
        # Use the same color as assigned in species_colors
        color = species_colors.get(species_name, 'gray')
        ax_species.plot(frame_times, probs, label=species_name, 
                       color=color, linewidth=2, alpha=0.8)
    
    ax_species.set_ylabel('Probability')
    ax_species.set_ylim(0, 1)
    ax_species.set_xlim(0, duration)
    # Don't add legend here - will use unified legend at bottom
    ax_species.grid(True, alpha=0.3)
    ax_species.set_xlabel('')
    
    # ============= ROW 3: GROUND TRUTH =============
    ax_gt = fig.add_subplot(gs[2], sharex=ax_spec)
    ax_gt.set_ylim(0, 1)
    ax_gt.set_ylabel('Ground\nTruth', fontsize=10, rotation=0, ha='right', va='center')
    ax_gt.set_yticks([])
    ax_gt.set_xlim(0, duration)
    
    for _, row in selection_table.iterrows():
        start = row["Begin Time (s)"]
        end = row["End Time (s)"]
        species = row[annotation_column]
        width = end - start
        
        color = species_colors.get(species, (0.5, 0.5, 0.5, 0.7))
        rect = mpatches.Rectangle((start, 0.1), width, 0.8, 
                                  linewidth=1, edgecolor='black', 
                                  facecolor=color, alpha=0.7)
        ax_gt.add_patch(rect)
    
    ax_gt.grid(True, axis='x', alpha=0.3)
    ax_gt.set_xlabel('')
    
    # ============= ROW 4: PREDICTIONS =============
    ax_pred = fig.add_subplot(gs[3], sharex=ax_spec)
    ax_pred.set_ylim(0, 1)
    ax_pred.set_ylabel('Prediction', fontsize=10, rotation=0, ha='right', va='center')
    ax_pred.set_yticks([])
    ax_pred.set_xlim(0, duration)
    
    for start, end, species in pred_events:
        width = end - start
        color = species_colors.get(species, (0.5, 0.5, 0.5, 0.7))
        rect = mpatches.Rectangle((start, 0.1), width, 0.8, 
                                  linewidth=1, edgecolor='black', 
                                  facecolor=color, alpha=0.7)
        ax_pred.add_patch(rect)
    
    ax_pred.grid(True, axis='x', alpha=0.3)
    ax_pred.set_xlabel('Time (s)')
    
    # ============= UNIFIED LEGEND =============
    # Create legend for animal detection line and all species
    legend_species = sorted([s for s in all_species if s in (gt_species | pred_species | top_5_species)])
    legend_handles = []
    
    # Add animal detection as first item in legend
    from matplotlib.lines import Line2D
    animal_line = Line2D([0], [0], color='black', linewidth=4, label='Animal Detection')
    legend_handles.append(animal_line)
    
    # Add species patches
    if legend_species:
        species_patches = [
            mpatches.Patch(facecolor=species_colors[s], edgecolor='black', 
                          label=s, alpha=0.7)
            for s in legend_species
        ]
        legend_handles.extend(species_patches)
    
    # Place legend below the plot
    if legend_handles:
        fig.legend(handles=legend_handles, loc='lower center', 
                  ncol=min(6, len(legend_handles)), 
                  fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    # ============= STATISTICS =============
    n_gt = len(selection_table)
    n_pred = len(pred_events)
    stats_text = f'Ground Truth Events: {n_gt}  |  Predicted Events: {n_pred}'
    fig.text(0.5, 0.96, stats_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualization: {output_path}")
