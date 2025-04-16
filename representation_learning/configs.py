"""
configs.py
~~~~~~~~~~
Canonical **Pydantic v2** data‑classes for training‑run YAML files.

The schema is deliberately strict (`extra='forbid'`) so that typos in
configuration files raise immediately.

Usage
-----
>>> from pathlib import Path, yaml
>>> from configs import RunConfig
>>> cfg = RunConfig.model_validate(yaml.safe_load(Path("run.yml").read_text()))
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path
import yaml

# --------------------------------------------------------------------------- #
#  3rd‑party imports
# --------------------------------------------------------------------------- #
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.dataclasses import dataclass  # validates dataclass fields
from dataclasses import field as dc_field

# --------------------------------------------------------------------------- #
#  Training‑level hyper‑parameters
# --------------------------------------------------------------------------- #

class TrainingParams(BaseModel):
    """Hyper‑parameters that control optimisation."""

    train_epochs: int = Field(..., ge=1, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., ge=1, description="Batch size for training")
    optimizer: Literal["adamw", "adam"] = Field("adamw", description="Optimizer to use")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay for regularisation")

    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    model_config = ConfigDict(extra="forbid")

# --------------------------------------------------------------------------- #
#  Data‑augmentation sections
# --------------------------------------------------------------------------- #

class NoiseAugment(BaseModel):
    kind: Literal["noise"] = "noise"
    noise_dirs: List[str]
    snr_db_range: Tuple[int, int] = Field(..., min_length=2, max_length=2)
    augmentation_prob: float = Field(..., ge=0, le=1)

    model_config = ConfigDict(extra="forbid")


class MixupAugment(BaseModel):
    kind: Literal["mixup"] = "mixup"
    alpha: float = Field(..., gt=0)
    augmentation_prob: float = Field(..., ge=0, le=1)

    model_config = ConfigDict(extra="forbid")


Augment = Union[NoiseAugment, MixupAugment]


# --------------------------------------------------------------------------- #
#  Audio & model configuration
# --------------------------------------------------------------------------- #

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: Optional[int] = None
    win_length: Optional[int] = None
    window: Literal["hann", "hamming"] = "hann"
    n_mels: int = 128
    representation: Literal["spectrogram", "mel_spectrogram", "raw"] = "mel_spectrogram"
    normalize: bool = True
    target_length: Optional[int] = None
    window_selection: Literal["random", "center"] = "random"

    model_config = ConfigDict(extra="forbid")


class ModelSpec(BaseModel):
    """All parameters required to *instantiate* the network."""

    name: str
    pretrained: bool = True
    device: str = "cuda"
    audio_config: Optional[AudioConfig] = None

    model_config = ConfigDict(extra="forbid")

# --------------------------------------------------------------------------- #
#  Dataset‑filtering helpers
# --------------------------------------------------------------------------- #

class FilterConfig(BaseModel):
    property: str
    values: List[str]
    operation: Literal["include", "exclude"] = "include"


class SubsampleConfig(BaseModel):
    property: str
    operation: Literal["subsample"] = "subsample"
    ratios: Dict[str, float] = dc_field(default_factory=dict)

TransformCfg = Union[FilterConfig, SubsampleConfig]


class DataConfig(BaseModel):
    dataset_name: str
    label_column: str
    label_type: Literal["supervised", "self-supervised"]
    transformations: Optional[List[TransformCfg]] = None        # <- changed
    read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)

# --------------------------------------------------------------------------- #
#  Top‑level run‑configuration
# --------------------------------------------------------------------------- #

class RunConfig(BaseModel):
    """Everything needed for a single *training run*."""

    # required
    model_spec: ModelSpec
    training_params: TrainingParams
    dataset_config: str

    # optional / misc
    preprocessing: Optional[str] = None
    sr: int = 16000
    logging: Literal["mlflow", "wandb"] = "mlflow"

    augmentations: List[Augment] = Field(default_factory=list)
    loss_function: Literal["cross_entropy", "bce"]

    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    run_name: Optional[str] = None
    wandb_project: str = "audio‑experiments"

    # ------------------------------
    # custom pre‑processing of augments
    # ------------------------------
    @field_validator("augmentations", mode="before")
    @classmethod
    def _flatten_augments(cls, raw: Any) -> Any:
        """Convert YAML single‑key mapping style into flat dicts.

        YAML allows:
            - noise: {noise_dirs: [...], snr_db_range: [-5, 20], augmentation_prob: 0.5}
        This turns into {kind:"noise", noise_dirs: [...], ...} so the discriminated
        union resolves correctly.
        """
        if not raw:
            return []

        processed: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and len(item) == 1:
                aug_type, params = next(iter(item.items()))
                params = params or {}
                params["kind"] = aug_type
                processed.append(params)
            else:
                processed.append(item)
        return processed

    model_config = ConfigDict(extra="forbid")

# --------------------------------------------------------------------------- #
#  Convenience loader
# --------------------------------------------------------------------------- #

def load_config(path: str | Path, config_type = "run") -> RunConfig:
    """Read YAML at *path*, validate, and return a **RunConfig** instance."""

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    
    if config_type == "run":
        return RunConfig.model_validate(raw)
    elif config_type == "data":
        return DataConfig.model_validate(raw)
    else:
        raise NotImplementedError("Can only load from run config or data config.")
