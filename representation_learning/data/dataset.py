from __future__ import annotations

import multiprocessing
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import RunConfig, load_config
from representation_learning.data.audio_utils import (
    pad_or_window,  # type: ignore
)


# --------------------------------------------------------------------------- #
#  Collater
# --------------------------------------------------------------------------- #
class Collater:
    """
    Combines samples into a batch, ensuring every audio clip has the same
    length (`audio_max_length`) by truncating or zero‑padding as needed.
    """

    def __init__(
        self,
        audio_max_length_seconds: int,
        sr: int,
        window_selection: str = "random",
        keep_text: bool = False,
        preprocessor: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.window_selection = window_selection
        self.keep_text = keep_text
        self.preprocessor = preprocessor
        self.sr = sr
        self.device = device

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audios, masks, labels, text_labels = [], [], [], []

        for item in batch:
            wav = torch.as_tensor(item["raw_wav"])  # (T,)
            wav, pad_mask = pad_or_window(
                wav, self.audio_max_length_seconds * self.sr, self.window_selection
            )
            audios.append(wav)
            masks.append(pad_mask)
            labels.append(item["label"])
            if self.keep_text:
                text_labels.append(item["text_label"])

        # Keep tensors on CPU for pinning
        audio_tensor = torch.stack(audios)  # [B, T] float32
        mask_tensor = torch.stack(masks)  # [B, T] bool
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }


def build_dataloaders(
    cfg: RunConfig, device: str = "cpu"
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration containing dataset and training parameters
    device : str
        Device to use for data loading

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)

    # Load dataset configuration
    data_config = load_config(cfg.dataset_config, config_type="data")

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=False,  # TEMP: for testing speed
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True,
    )

    # Create collater
    collate_fn = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        device=device,
    )

    # Create dataloaders
    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    val_dl = DataLoader(
        ds_eval,
        batch_size=cfg.training_params.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return train_dl, val_dl


### this is temporary, it should be replaced by get_dataset or get_benchmark (with the appropriate split)
def get_benchmark_dataset_dummy(
    data_config: Any,
    csv_path: str,
    preprocessor: Optional[Callable] = None,
    validation: bool = False
) -> AudioDataset:
    """
    Dataset entry point that supports both local paths, with transformations.
    
    1. Loads metadata CSV 
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.
    """
    
    # Check if the dataset CSV path is a gs:// path
    df = pd.read_csv(csv_path)

    # Apply transformations if specified
    if hasattr(data_config, 'transformations') and data_config.transformations:
        transforms = build_transforms(data_config.transformations)
        for transform in transforms:
            df = transform(df)

    return AudioDatasetBenchmark(
        metadata_df=df,
        data_config=data_config,
        preprocessor=preprocessor,
    )
    
#### this is temporary, it should be integrated into the build_dataloaders function
def build_evaluation_dataloaders(run_config, model_spec, data_config, device="cpu"):
    """
    Build training and validation dataloaders from configuration.
    
    Args:
        run_config: Run configuration
        model_spec: Model configuration
        data_config: Data configuration
        device: Device to use for data loading
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_benchmark_dataset_dummy(
        data_config=data_config,
        csv_path=Path(data_config.train_path),
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True #TEMP: for testing speed
    )
    ds_eval = get_benchmark_dataset_dummy(
        data_config=data_config,
        csv_path=Path(data_config.valid_path),
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True
    )
    
    # Create collater
    collate_fn = Collater(
        audio_max_length_seconds=model_spec.audio_config.target_length,
        sr=model_spec.audio_config.sample_rate,
        window_selection=model_spec.audio_config.window_selection
    )
    
    # Create dataloaders
    train_dl = DataLoader(
        ds_train,
        batch_size=run_config.training_params.batch_size,
        shuffle=True,
        num_workers=run_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )
    
    val_dl = DataLoader(
        ds_eval,
        batch_size=run_config.training_params.batch_size,
        shuffle=False,
        num_workers=run_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )
    
    return train_dl, val_dl

class AudioDatasetBenchmark(Dataset):
    """
    Reads metadata from a CSV, loads audio, and yields a sample dict.
    
    Expected columns in the CSV:
    * 'filepath'  : str – path to the audio file on disk or a gs:// path.
    * <label_col> : str – value used for the target (e.g. species name).
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_config: Any,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_config = data_config
        self.preprocessor = preprocessor

        self.audio_path_col = data_config.audio_path_col
        self.label_col = data_config.label_column

        # Build a label → index mapping for numeric targets
        unique_labels = sorted(self.metadata[self.label_col].unique())
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        if path_str.startswith("gs://"):
            if GSPath is None:
                raise ImportError("cloudpathlib is required to handle gs:// paths.")
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        # Open the audio file. Using the .open('rb') method works for both local and GSPath objects.
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        return {
            "raw_wav": audio.astype(np.float32),
            "text_label": row[self.label_col],
            "label": self.label2idx[row[self.label_col]],
            "path": str(audio_path),
        }

