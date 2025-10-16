"""
Evaluate combined detector-classifier model.

This script combines:
1. PretrainedSED (BEATs) for frame-level animal detection at 25 Hz
2. Perch classifier for species classification with sliding windows
3. Combines detection × classification for final predictions
"""

from representation_learning.models.perch import PerchModel
import torch
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from zeroshot_detection_eval.data.dataloader import DataloaderBuilder
from zeroshot_detection_eval.eval.evaluation import Scorer
from esp_data.io import filesystem
import os 
from tqdm import tqdm

# Import visualization function
from visualize_detector_classifier import plot_visualization

# Import from installed packages
from PretrainedSED.data_util.audioset_classes import as_strong_xc_classes, as_strong_train_classes
from PretrainedSED.models.beats.BEATs_wrapper import BEATsWrapper
from PretrainedSED.models.prediction_wrapper import PredictionsWrapper

# ============= CONFIGURATION =============
DATASET = "xeno_canto_annotated_jeantet_2023"
ANNOTATION_COLUMN = "Species" 

# Perch classifier parameters (configurable)
PERCH_WINDOW = 2      # seconds
PERCH_HOP = 1         # seconds
PERCH_BATCH_SIZE = 4
PERCH_N_WORKERS = 4

# PretrainedSED detector parameters (fixed by model architecture)
SED_WINDOW = 10.0       # Required window size for BEATs model
SED_HOP = 10.0          # seconds
SED_N_WORKERS = 0
SED_LOWPASS_CUTOFF = 2.0  # Hz - Low-pass filter cutoff frequency for smoothing animal detection

# Output frame rate from ptsed (fixed by model architecture)
SR_SED = 16000  # All PretrainedSED models require 16kHz
SR_PERCH = 32000  # Perch model requires 32kHz
FRAME_RATE = 250 / SED_WINDOW  # ptsed outputs 250 frames per 10s clip

# Visualization configuration
VIZ_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]  # Thresholds for visualization
VIZ_OUTPUT_DIR = f".results/vis_combined_{PERCH_WINDOW}_{PERCH_HOP}"  # Output directory for plots

out_fp_combined = f".results/eval_combined_{PERCH_WINDOW}_{PERCH_HOP}.yaml"
out_fp_perch_only = f".results/eval_perch_only_{PERCH_WINDOW}_{PERCH_HOP}.yaml"


# ============= HELPER FUNCTIONS =============

def get_animal_class_indices():
    """Get indices of animal-related classes in AudioSet."""
    animal_class_indices = [
        i for i, cls in enumerate(as_strong_train_classes) 
        if cls in as_strong_xc_classes
    ]
    return animal_class_indices

def apply_sed_detector(
    sed_dataloader,
    sed_model, 
    animal_class_indices: list,
    device: torch.device,
    lowpass_cutoff: float,
    frame_rate: float
) -> np.ndarray:
    """
    Apply PretrainedSED detector to audio and extract animal detection probabilities.
    
    Parameters
    ----------
    sed_dataloader : torch.utils.data.DataLoader
        DataLoader providing audio chunks at 16kHz
    sed_model : torch.nn.Module
        PretrainedSED model
    animal_class_indices : list
        Indices of animal-related classes in AudioSet
    device : torch.device
        Device to run model on
    lowpass_cutoff : float
        Cutoff frequency in Hz for low-pass filter
    frame_rate : float
        Frame rate of the output (frames per second)
        
    Returns
    -------
    np.ndarray
        Animal detection probabilities (low-pass filtered), shape (total_frames,)
    """
    all_animal_probs = []
    
    for batch in sed_dataloader:
        # batch: (batch_size, num_samples)
        batch = batch.to(device)
        
        with torch.no_grad():
            # BEATs preprocessing: convert to mel spectrogram
            mel = sed_model.mel_forward(batch)  # (batch, 1, time, freq)
            y_strong, y_weak = sed_model(mel)  # Returns (strong, weak) tuple
            # y_strong: (batch, num_audioset_classes, num_time_frames)
            y_strong = torch.sigmoid(y_strong)

        # Extract animal class probabilities and take max across animal classes
        animal_probs = y_strong[:, animal_class_indices, :]  # (1, num_animal_classes, num_time_frames)
        animal_detection = animal_probs.max(dim=1)[0]      # (1, num_time_frames)
        
        all_animal_probs.append(animal_detection.cpu().numpy())
    
    # Concatenate all chunks and flatten to single sequence
    animal_probs = np.concatenate(all_animal_probs, axis=0)  # (num_chunks, frames_per_chunk)
    animal_probs = animal_probs.flatten()  # (total_frames,)
    
    # Apply low-pass filter to smooth the animal detection probabilities
    # Design a low-pass Butterworth filter
    nyquist = frame_rate / 2  # Nyquist frequency
    if lowpass_cutoff < nyquist:
        # Normalize cutoff frequency by Nyquist frequency
        normalized_cutoff = lowpass_cutoff / nyquist
        b, a = butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)
        # Apply zero-phase filter (forward and backward)
        animal_probs = filtfilt(b, a, animal_probs)
        # Clip to [0, 1] range in case filter introduces small overshoots
        animal_probs = np.clip(animal_probs, 0.0, 1.0)
    
    return animal_probs


def align_windows_to_frames(
    window_preds: np.ndarray,
    window_size: float,
    hop_size: float,
    num_frames: int
) -> np.ndarray:
    """
    Align sliding window predictions to frame-level resolution by averaging.
    
    Parameters
    ----------
    window_preds : np.ndarray
        Predictions from sliding windows, shape (num_windows, num_classes)
    window_size : float
        Duration of each window in seconds
    hop_size : float
        Hop size between windows in seconds
    num_frames : int
        Total number of frames to output
        
    Returns
    -------
    np.ndarray
        Frame-level predictions, shape (num_frames, num_classes)
    """
    num_windows, num_classes = window_preds.shape
    
    # Initialize output and coverage counter
    species_probs = np.zeros((num_frames, num_classes))
    coverage_count = np.zeros(num_frames)
    
    # For each window, add its predictions to all frames it covers
    for window_idx in range(num_windows):
        window_start_time = window_idx * hop_size
        window_end_time = window_start_time + window_size
        
        # Calculate which frames this window covers
        first_frame = int(window_start_time * FRAME_RATE)
        last_frame = min(int(window_end_time * FRAME_RATE), num_frames - 1)

        # Add predictions to all covered frames
        species_probs[first_frame:last_frame+1] += window_preds[window_idx]
        coverage_count[first_frame:last_frame+1] += 1
    
    # Average by dividing by coverage count
    species_probs /= coverage_count[:, None]
    
    return species_probs


def load_species_label_mapping(dataset: str):
    """Load and create mapping from eBird codes to scientific names."""
    # Get absolute path for perch_labels.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perch_labels_path = os.path.join(script_dir, 'perch_labels.csv')
    output_labels = pd.read_csv(perch_labels_path)['ebird2021_code'].tolist()

    if dataset == "powdermill":
        info_fp = "Clements-v2024-October-2024-rev.csv"
        if not os.path.exists(info_fp):
            fs = filesystem("gcs")
            gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/Clements-v2024-October-2024-rev.csv"
            fs.get(gcp_path, info_fp)

        info_df = pd.read_csv(info_fp)
        info_df = info_df[~pd.isna(info_df["scientific name"])]
        
    elif dataset == "wabad":
        info_fp = "eBird_Taxonomy_v2021.csv"
        if not os.path.exists(info_fp):
            fs = filesystem("gcs")
            gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/eBird_Taxonomy_v2021.csv"
            fs.get(gcp_path, info_fp)

        info_df = pd.read_csv(info_fp)
        info_df = info_df[~pd.isna(info_df["SCI_NAME"])]
        info_df['species_code'] = info_df["SPECIES_CODE"]
        info_df['scientific name'] = info_df["SCI_NAME"]
    
    elif dataset == "xeno_canto_annotated_jeantet_2023":
        # For xeno-canto, labels are already scientific names
        # We still need to map Perch eBird codes to scientific names for matching
        info_fp = "eBird_Taxonomy_v2021.csv"
        if not os.path.exists(info_fp):
            fs = filesystem("gcs")
            gcp_path = "gs://esp-ml-datasets/wabad/v0.1.0/eBird_Taxonomy_v2021.csv"
            fs.get(gcp_path, info_fp)

        info_df = pd.read_csv(info_fp)
        info_df = info_df[~pd.isna(info_df["SCI_NAME"])]
        info_df['species_code'] = info_df["SPECIES_CODE"]
        info_df['scientific name'] = info_df["SCI_NAME"]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets are: powdermill, wabad, xeno_canto_annotated_jeantet_2023")

    output_species_labels = []
    print("Converting eBird codes to scientific names...")
    for ebird_label in tqdm(output_labels):
        df_sub = info_df[info_df['species_code'] == ebird_label]
        
        if dataset == "powdermill":
            if ebird_label == "haiwoo":
                scientific_name = 'Leuconotopicus villosus'
                output_species_labels.append(scientific_name)
                continue
            if ebird_label == "reevir":
                scientific_name = 'Vireo olivaceus'
                output_species_labels.append(scientific_name)
                continue

        if dataset == "wabad":
            correction_dict = {}
            if ebird_label in correction_dict.keys():
                scientific_name = correction_dict[ebird_label]
                output_species_labels.append(scientific_name)
                continue

        if len(df_sub) == 0:
            scientific_name = ebird_label
        else:
            scientific_name = df_sub['scientific name'].tolist()[0]
        output_species_labels.append(scientific_name)

    # Add bonus columns for WABAD dataset
    if dataset == "wabad":
        bonus_columns = [
            ("bkhbat1", "Batis minor"),
            ("subwar1", "Curruca cantillans"),
            ("leswhi1", "Curruca curruca"),
            ("subwar1", "Curruca iberiae"),
            ("grnjay1", "Cyanocorax yncas"),
            ("ruwant4", "Herpsilochmus frater"),
            ("blewhe1", "Oenanthe hispanica"),
            ("blcapa2", "Oreolais rufogularis"),
            ("grnwoo1", "Picus viridis"),
            ("gowbar1", "Psilopogon chrysopogon"),
            ("reevir", "Vireo olivaceus"),
            ("butwoo2", "Xiphorhynchus guttatus")
        ]
    else:
        bonus_columns = []

    return output_labels, output_species_labels, bonus_columns


# ============= MAIN =============

def main():
    # Load perch
    perch_model = PerchModel(0)
    perch_model.eval()
    
    # Load PretrainedSED 
    device = (torch.device('cuda') if torch.cuda.is_available()
               else torch.device('cpu'))
    
    beats = BEATsWrapper()
    sed_model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
    sed_model.eval()
    sed_model.to(device)
    print(f"  Using device: {device}")
    
    animal_class_indices = get_animal_class_indices()
    
    # Load datasets - one for SED at 16kHz, one for Perch at 32kHz
    sed_dl = DataloaderBuilder(
        DATASET, SED_WINDOW, SED_HOP, batch_size=1, 
        num_workers=SED_N_WORKERS, sr=SR_SED
    )
    
    perch_dl = DataloaderBuilder(
        DATASET, PERCH_WINDOW, PERCH_HOP, PERCH_BATCH_SIZE, 
        PERCH_N_WORKERS, sr=SR_PERCH
    )
    
    # Load label mappings
    output_labels, output_species_labels, bonus_columns = load_species_label_mapping(DATASET)
    
    # Check label coverage (use perch_dl's dataset as the reference)
    target_labels = perch_dl.ds.get_available_labels(ANNOTATION_COLUMN)
    
    for species in target_labels:
        if species not in output_species_labels and \
           species not in [x[1] for x in bonus_columns]:
            print(f"Warning: Species not in output labels: {species}")

    # Initialize scorers - one for combined (detection x classification), one for perch-only
    S_combined = Scorer(temp_dir=f".results/eval_combined_{PERCH_WINDOW}_{PERCH_HOP}")
    S_perch_only = Scorer(temp_dir=f".results/eval_perch_only_{PERCH_WINDOW}_{PERCH_HOP}")
    
    # Create visualization output directory
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
    
    # Process each audio file
    for file_idx in range(len(perch_dl)):
        print(f"\n{'='*60}")
        print(f"Processing file {file_idx + 1}/{len(perch_dl)}")
        print(f"{'='*60}")
        if file_idx !=0:
            continue
    
        # -------- STEP 1: PretrainedSED for Animal Detection --------
        
        # Get SED dataloader (16kHz audio, 10s windows)
        sed_data = sed_dl[file_idx]
        sed_dataloader = sed_data['dataloader']
        
        animal_probs = apply_sed_detector(
            sed_dataloader, sed_model, animal_class_indices,
            device, SED_LOWPASS_CUTOFF, FRAME_RATE
        )

        # Calculate correct output frame count and crop, removing padding
        # Use Perch audio (32kHz) to get the true audio duration
        perch_audio = perch_dl.ds[file_idx]["audio"]
        audio_duration_sec = len(perch_audio) / SR_PERCH
        num_frames = int(np.ceil(audio_duration_sec * FRAME_RATE))
        animal_probs = animal_probs[:num_frames]
        
        # -------- STEP 2: Perch for Species Classification --------
        
        # Get Perch dataloader (32kHz audio, 5s windows with 2.5s hop)
        perch_data = perch_dl[file_idx]
        perch_dataloader = perch_data['dataloader']

        preds = []
        for batch in tqdm(perch_dataloader):
            with torch.no_grad():
                batchpreds = perch_model(batch).detach().numpy()
                preds.append(batchpreds)

        preds = np.concatenate(preds,axis=0)
        preds_df = pd.DataFrame(preds, columns=output_species_labels)
        if DATASET == "wabad":
            bonus_preds_df = pd.DataFrame(preds, columns=output_labels)
            bonus_preds_df = bonus_preds_df[[x[0] for x in bonus_columns]]
            bonus_preds_df.columns = [x[1] for x in bonus_columns]
            preds_df = pd.concat([preds_df, bonus_preds_df], axis =1)
        preds_df = preds_df[target_labels]
        preds = preds_df.to_numpy()
        preds = 1. / (1. + np.exp(-preds))

        # -------- STEP 3: Combine Detection x Classification --------
        species_probs = align_windows_to_frames(
            preds, PERCH_WINDOW, PERCH_HOP, num_frames
        )

        final_scores = species_probs * animal_probs[:, None]

        # Update scorers
        S_combined.update(final_scores, target_labels, FRAME_RATE, perch_data['selection_table'], ANNOTATION_COLUMN)
        S_perch_only.update(species_probs, target_labels, FRAME_RATE, perch_data['selection_table'], ANNOTATION_COLUMN)

        # -------- STEP 6: Create Visualizations --------
        # Only visualize every 50th file to save disk space
        if file_idx % 50 == 0:
            file_output_dir = os.path.join(VIZ_OUTPUT_DIR, f"{file_idx:03d}")
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Create visualizations at different thresholds
            for threshold in VIZ_THRESHOLDS:
                output_path = os.path.join(file_output_dir, 
                                           f"{file_idx:03d}_thresh{threshold:.2f}.png")
                
                plot_visualization(
                    audio=perch_audio,
                    sr=SR_PERCH,
                    species_probs=species_probs,
                    output_species_labels=target_labels,
                    animal_probs=animal_probs,
                    selection_table=perch_data['selection_table'],
                    annotation_column=ANNOTATION_COLUMN,
                    final_scores=final_scores,
                    frame_rate=FRAME_RATE,
                    threshold=threshold,
                    output_path=output_path
                )

           
    # Compute final metrics
    S_combined.compute_scores(output_fp=out_fp_combined, delete_temp=False, num_jobs=12)
    S_perch_only.compute_scores(output_fp=out_fp_perch_only, delete_temp=False, num_jobs=12)


if __name__ == "__main__":
    main()
