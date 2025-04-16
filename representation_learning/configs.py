"""
configs.py
~~~~~~~~~~
Dataclasses for the training‑run YAML.

Usage
-----
>>> from pathlib import Path
>>> import yaml
>>> from configs import RunConfig
>>>
>>> cfg_dict = yaml.safe_load(Path("run.yml").read_text())
>>> cfg      = RunConfig.model_validate(cfg_dict)   # validated & parsed
"""

from __future__ import annotations

from typing import List, Tuple, Union, Literal, Optional, Any, Dict
from pathlib import Path
import yaml
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator


class TrainingParams(BaseModel):
    train_epochs: int = Field(..., ge=1, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., ge=1, description="Batch size for training")
    optimizer: Literal["adamw", "adam"] = Field(default="adamw", description="Optimizer to use")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay for regularization")
    amp: bool = False
    amp_dtype: str = "bf16"  # or "fp16"

    def __post_init__(self):
        """Validate parameters."""
        if self.amp_dtype not in ["bf16", "fp16"]:
            raise ValueError("amp_dtype must be either 'bf16' or 'fp16'")

class NoiseAugment(BaseModel):
    kind: Literal["noise"] = Field(default="noise", description="Type of augmentation")
    noise_dirs: List[str] = Field(..., description="Directories containing noise files")
    snr_db_range: Tuple[int, int] = Field(..., min_items=2, max_items=2, description="SNR range in dB")
    augmentation_prob: float = Field(..., ge=0, le=1, description="Probability of applying this augmentation")

class MixupAugment(BaseModel):
    kind: Literal["mixup"] = Field(default="mixup", description="Type of augmentation")
    alpha: float = Field(..., gt=0, description="Mixup alpha parameter")
    augmentation_prob: float = Field(..., ge=0, le=1, description="Probability of applying this augmentation")

# Union of the augmentation types
Augment = Union[NoiseAugment, MixupAugment]

class AudioConfig(BaseModel):
    """Configuration for audio processing."""
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    n_fft: int = Field(default=2048, description="Number of FFT bins")
    hop_length: Optional[int] = Field(default=None, description="Hop length between STFT windows")
    win_length: Optional[int] = Field(default=None, description="Window length for STFT")
    window: Literal["hann", "hamming"] = Field(default="hann", description="Window function type")
    n_mels: int = Field(default=128, description="Number of mel bands")
    representation: Literal["spectrogram", "mel_spectrogram", "raw"] = Field(
        default="mel_spectrogram",
        description="Type of audio representation to use"
    )
    normalize: bool = Field(default=True, description="Whether to normalize the output")
    target_length: Optional[int] = Field(default=None, description="Target length in samples for padding/windowing")
    window_selection: Literal["random", "center"] = Field(default="random", description="Method for selecting windows")

class ModelConfig(BaseModel):
    """Configuration for model instantiation."""
    name: str = Field(..., description="Name of the model to use")
    pretrained: bool = Field(default=True, description="Whether to use pretrained weights")
    device: str = Field(default="cuda", description="Device to run on (cuda/cpu)")
    audio_config: Optional[AudioConfig] = Field(default=None, description="Audio processing configuration")

@dataclass
class FilterConfig:
    """Configuration for filtering data based on property values."""
    property: str
    values: List[str]
    operation: str = "include"  # "include" or "exclude"

@dataclass
class SubsampleConfig:
    """Configuration for subsampling data based on property ratios."""
    property: str
    operation: str = "subsample"
    ratios: Dict[str, float] = field(default_factory=dict)

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_path: Union[str, Path]
    label_column: str
    label_type: str  # "supervised" or "unsupervised"
    transformations: Optional[List[Dict[str, Union[FilterConfig, SubsampleConfig]]]] = None
    read_csv_kwargs: Dict = field(default_factory=dict)

class RunConfig(BaseModel):
    # Core
    model_config: ModelConfig = Field(..., description="Model configuration")
    dataset_config: str = Field(..., description="Path to dataset configuration")
    preprocessing: Optional[str] = Field(None, description="Optional preprocessing step")
    sr: int = Field(default=16000, description="Target sample rate")
    logging: Literal["mlflow", "wandb"] = Field(default="mlflow", description="Logging framework to use")
    training_params: TrainingParams = Field(..., description="Training parameters")
    augmentations: List[Augment] = Field(..., description="List of data augmentations")
    loss_function: Literal["cross_entropy", "bce"] = Field(..., description="Loss function to use")
    
    # Additional parameters found in codebase
    device: str = Field(default="cuda", description="Device to run on (cuda/cpu)")
    seed: int = Field(default=42, description="Random seed")
    num_workers: int = Field(default=4, description="Number of data loader workers")
    run_name: Optional[str] = Field(None, description="Name of the run for logging")
    wandb_project: str = Field(default="audio-experiments", description="Weights & Biases project name")

    @field_validator("augmentations", mode="before")
    @classmethod
    def _parse_augments(cls, raw_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert the YAML representation (each item is a *single‑key* dict)
        into proper Augment objects.
        """
        parsed: List[Dict[str, Any]] = []

        for item in raw_list:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(
                    "Each augmentation entry must be a one‑key mapping, "
                    "e.g.  - mixup: {alpha: 0.4, augmentation_prob: 0.3}"
                )

            aug_type, params = next(iter(item.items()))
            params = params or {}  # allow e.g. `- mixup`

            # Add the kind field to help with discriminated union
            if isinstance(params, dict):
                params["kind"] = aug_type

            parsed.append(params)

        return parsed


def load_config(path: str | Path) -> RunConfig:
    """Read YAML at *path*, validate, and return a RunConfig instance."""

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    try:
        cfg = RunConfig.model_validate(raw)
    except Exception as e:
        print("Config validation failed:", e)
        raise

    return cfg
