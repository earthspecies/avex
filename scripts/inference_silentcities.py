import os
import argparse

from tqdm import tqdm
from representation_learning import load_model
import torch
import torchaudio  # noqa: F401  # (kept in case model uses it internally)
from torch.utils.data import Dataset, DataLoader
from esp_data.io import read_audio
import librosa
import pandas as pd
import numpy as np

MODEL_SR = 32000
TARGET_SEC = 5.

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

label_to_idx = pd.read_json(
    "gs://representation-learning/models/v1/label_map.json",
    typ="series",
).to_dict()
idx_to_label = {label_to_idx[x]: x for x in label_to_idx}

model = load_model(
    "sl_beats_all",
    checkpoint_path="gs://representation-learning/models/v1/beats_32khz.pt",
    num_classes=None,
)

# -----------------------
# Load + filter metadata
# -----------------------
metadata = pd.read_csv('gs://fewshot/data_large_clean/silentcities_sampled_big_info.csv')

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
        Sample rate, e.g. 16000

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


class ClipDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame):
        """
        Parameters
        ----------
        metadata : pandas.DataFrame
            Must contain a column 'audio_fp'.
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

        audio_fp = row["audio_fp"]
        try:
            waveform, sr = read_audio(audio_fp)  # hack: missing / corrupt files get skipped
            qf_exclude = False
        except:
            qf_exclude = True
            waveform = np.zeros((int(MODEL_SR * TARGET_SEC),), )
            sr = MODEL_SR


        if sr != MODEL_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=MODEL_SR)
            sr = MODEL_SR

        waveform = torch.tensor(waveform)

        waveform = pad_and_crop(waveform, TARGET_SEC, MODEL_SR)

        return df_idx, waveform, qf_exclude  # return df index so we can write back later

def logits_to_animalfilter_exclude(logit_vec: np.ndarray, idx_to_label: dict) -> str:
    """
    Looks at all indices where logit_vec >= 0; returns exclude=True if any of them are animals 
    (quick heuristic: if classifier output is a two word name; ie binomial name)
    """
    idxs = np.where(logit_vec >= 0)[0]          # indices with logit >= 0
    labels = [idx_to_label[i] for i in idxs]    # map each index to label
    for label in labels:
        if len(label.split(' ')) == 2:
            if ',' not in label:
                return True
    return False

# -------------------------------
# Inference + writeback into shard
# -------------------------------

model = model.to(device)
model.eval()

dataset = ClipDataset(shard_metadata)
loader = DataLoader(
    dataset,
    batch_size=64,   # adjust as needed
    shuffle=False,
    num_workers=12,  # adjust for your system
)

# Make sure the columns exist for this shard
for col in [
    "preds_csv"
]:
    if col not in shard_metadata.columns:
        shard_metadata[col] = None

with torch.no_grad():
    for df_indices, batch_waveforms, qf_excludes in tqdm(loader):
        batch_waveforms = batch_waveforms.to(device)

        logits = model(batch_waveforms)  # expected shape [B, num_classes]

        logits_cpu = logits.detach().cpu()
        logits_np = logits_cpu.numpy()

        # Write back to shard_metadata DataFrame
        for df_idx, logit_vec, qf_exclude in zip(df_indices, logits_np, qf_excludes):
            df_idx = int(df_idx)

            animalfilter_exclude = logits_to_animalfilter_exclude(logit_vec, idx_to_label)
            animalfilter_exclude = animalfilter_exclude or qf_exclude
            shard_metadata.at[df_idx, "animalfilter_exclude"] = animalfilter_exclude

# -----------------------
# Save shard-only output
# -----------------------
out_csv = (
    "silentcities_sampled_big_with_classifications"
    f"_shard{SHARD}.csv"
)
shard_metadata.to_csv(out_csv)
os.system(
    "gsutil -m cp -r "
    f"{out_csv} "
    "gs://fewshot/data_large_clean"
)
print(f"Wrote shard {SHARD} to {out_csv} and copied to GCS.")
