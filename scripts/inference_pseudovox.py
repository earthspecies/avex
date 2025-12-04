import os
import argparse

from tqdm import tqdm

import torch

if torch.cuda.is_available():
    _ = torch.cuda.device_count()  # Check device count
    # Create a small tensor on CUDA to fully initialize the context
    # This prevents CUDA from becoming unavailable during subsequent imports
    try:
        _cuda_init_tensor = torch.zeros(1, device="cuda")
        del _cuda_init_tensor  # Clean up immediately
        torch.cuda.synchronize()  # Ensure CUDA operations complete
    except Exception:
        pass  # If CUDA tensor creation fails, continue anyway

from representation_learning import load_model

import torch
import torchaudio  # noqa: F401  # (kept in case model uses it internally)
from torch.utils.data import Dataset, DataLoader
from esp_data.io import read_audio
import librosa
import pandas as pd

MODEL_SR = 32000
TARGET_SEC = 1.0

# def pad_and_crop(audio: torch.Tensor, target_dur_sec: float, sr: int) -> torch.Tensor:
#     """
#     Right-pad or left-crop a 1-D audio tensor to `target_dur_sec`.

#     Parameters
#     ----------
#     audio : torch.Tensor
#         1-D tensor of shape [T]. If shape is [1, T], pass audio.squeeze(0).
#     target_dur_sec : float
#         Target duration in seconds.
#     sr : int
#         Sample rate.

#     Raises
#     -------
#     ValueError
#         If audio is not 1-D.

#     Returns
#     -------
#     torch.Tensor
#         1-D tensor of shape [target_len].
#     """
#     if audio.dim() != 1:
#         raise ValueError(f"Expected a 1-D tensor, got shape: {audio.shape}")

#     target_len = int(round(target_dur_sec * sr))
#     T = audio.shape[0]

#     # Case 1: Exact length
#     if T == target_len:
#         return audio

#     # Case 2: Pad on the right
#     if T < target_len:
#         out = torch.zeros(target_len, dtype=audio.dtype, device=audio.device)
#         out[:T] = audio
#         return out

#     # Case 3: Crop by keeping the leftmost samples
#     return audio[:target_len]

def pad_and_crop(audio: torch.Tensor, target_dur_sec: float, sr: int) -> torch.Tensor:
    """
    Center pad or center crop a 1-D audio tensor to `target_dur_sec`.

    Parameters
    ----------
    audio : torch.Tensor
        1-D tensor of shape [T]. If shape is [1, T], pass audio.squeeze(0).
    target_dur_sec : float
        Target duration in seconds, e.g. 3.0
    sr : int
        Sample rate, e.g. 32000

    Raises
    -------
    ValueError
        If not right shape

    Returns
    -------
    torch.Tensor
        1-D tensor of shape [target_len].
    """
    if audio.dim() != 1:
        raise ValueError(
            "Expected a 1-D tensor for `audio`. Got shape: {}".format(audio.shape)
        )

    target_len = int(round(target_dur_sec * sr))
    T = audio.shape[0]

    # Case 1: Exact length
    if T == target_len:
        return audio

    # Case 2: Pad (audio shorter)
    if T < target_len:
        pad_total = target_len - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        out = torch.zeros(target_len, dtype=audio.dtype, device=audio.device)
        out[pad_left : pad_left + T] = audio
        return out

    # Case 3: Crop (audio longer)
    extra = T - target_len
    crop_left = extra // 2
    crop_right = crop_left + target_len
    return audio[crop_left:crop_right]

class PseudovoxDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame):
        """
        Parameters
        ----------
        metadata : pandas.DataFrame
            Must contain a column 'pseudovox_audio_fp'.
        """
        self.metadata = metadata
        # Work with positional indices internally, but remember the DataFrame index
        self._df_index = self.metadata.index.to_list()

    def __len__(self):
        return len(self._df_index)

    def __getitem__(self, i):
        # Map from positional index -> actual DataFrame index
        df_idx = self._df_index[i]
        row = self.metadata.loc[df_idx]

        audio_fp = (
            "/home/benjamin_earthspecies_org/synthetic_data/animalspeak_pseudovox/"
            + os.path.basename(row["pseudovox_audio_fp"])
        )
        if os.path.exists(audio_fp):
            waveform, sr = read_audio(audio_fp)  # hack: missing files get skipped
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_fp}")

        if sr != MODEL_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=MODEL_SR)
            sr = MODEL_SR

        dur = len(waveform)/sr

        waveform = torch.tensor(waveform)

        waveform = pad_and_crop(waveform, TARGET_SEC, MODEL_SR)

        return df_idx, waveform, dur  # return df index so we can write back later


def main():

    # -----------------------
    # CLI args for sharding
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Which shard index to process.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=12,
        help="Total number of shards.",
    )
    args = parser.parse_args()
    SHARD = args.shard
    NUM_SHARDS = args.num_shards

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    label_to_idx = pd.read_json(
        "gs://representation-learning/models/v1/label_map.json",
        typ="series",
    ).to_dict()
    idx_to_label = {label_to_idx[x]: x for x in label_to_idx}

    model = load_model(
        "sl_beats_all",
        checkpoint_path="gs://representation-learning/models/v1/beats_32khz.pt",
        num_classes=None,
        device=device
    )

    # -----------------------
    # Load + filter metadata
    # -----------------------
    metadata = pd.read_csv(
        "gs://fewshot/data_large_clean/"
        "animalspeak_pseudovox_with_birdnet_with_qf_with_c_with_meta.csv"
    )
    metadata = metadata[~metadata["qf_exclude"]].copy().reset_index(drop=True)

    # -----------------------
    # Compute shard slice
    # -----------------------
    n = len(metadata)
    if NUM_SHARDS <= 0:
        raise ValueError(f"num_shards must be > 0, got {NUM_SHARDS}")

    shard_size = (n + NUM_SHARDS - 1) // NUM_SHARDS  # ceil division
    start = SHARD * shard_size
    end = min(start + shard_size, n)

    if start >= n:
        raise ValueError(
            f"Shard {SHARD} is empty: start={start}, n={n}, num_shards={NUM_SHARDS}"
        )

    shard_metadata = metadata.iloc[start:end].copy()
    print(
        f"Processing shard {SHARD}/{NUM_SHARDS} "
        f"with rows [{start}, {end}) out of {n} total "
        f"({len(shard_metadata)} rows in this shard)."
    )

    # -------------------------------
    # Inference + writeback into shard
    # -------------------------------

    model = model.to(device)
    model.eval()

    dataset = PseudovoxDataset(shard_metadata)
    loader = DataLoader(
        dataset,
        batch_size=1,   # adjust as needed
        shuffle=False,
        num_workers=12,  # adjust for your system
        pin_memory=True,
    )

    # Make sure the columns exist for this shard
    cols_to_add = [
        "custom_classifier_logit_for_species_scientific_gbif",
        "custom_classifier_logit_for_species_scientific_animalspeak",
        "Duration (s)"
    ]
    cols_to_add += [f"custom_classifier_logit_for_background_species_{i}_scientific_gbif" for i in range(10)]
    for col in cols_to_add:
        if col not in shard_metadata.columns:
            shard_metadata[col] = -16.0

    keys_to_consider = ["species_scientific_gbif", "species_scientific_animalspeak"] + [f"background_species_{i}_scientific_gbif" for i in range(10)]
    ii = 0
    
    with torch.no_grad():
        for df_indices, batch_waveforms, batch_durs in tqdm(loader):
            ii +=1
            batch_waveforms = batch_waveforms.to(device, non_blocking=True)

            logits = model(batch_waveforms)  # expected shape [B, num_classes]

            logits_cpu = logits.detach().cpu()
            logits_np = logits_cpu.numpy()
            # pred_indices = logits_cpu.argmax(dim=-1).numpy()  # [B] (kept if needed)

            # Write back to shard_metadata DataFrame
            for df_idx, logit_vec, dur in zip(df_indices, logits_np, batch_durs):
                df_idx = int(df_idx)
                shard_metadata.at[df_idx, "Duration (s)"] = float(dur)
                for key in keys_to_consider:
                    # we look at both the original animalspeak metadata, as well as the corrected gbif metadata,
                    # for better recall of the correct name
                    species_in_metadata = shard_metadata.at[df_idx, key]
                    pred_idx = label_to_idx.get(species_in_metadata, -1)
                    if pred_idx == -1:
                        pass # already set to -16.0
                    else:
                        shard_metadata.at[
                            df_idx, f"custom_classifier_logit_for_{key}"
                        ] = float(logit_vec[pred_idx])
            # if ii == 10:
            #     break

    # -----------------------
    # Save shard-only output
    # -----------------------
    out_csv = (
        "animalspeak_pseudovox_with_birdnet_with_qf_with_c_with_meta_with_classifications"
        f"_shard{SHARD}.csv"
    )
    shard_metadata.to_csv(out_csv)
    os.system(
        "gsutil -m cp -r "
        f"{out_csv} "
        "gs://fewshot/data_large_clean"
    )
    print(f"Wrote shard {SHARD} to {out_csv} and copied to GCS.")

if __name__ == "__main__":
    main()