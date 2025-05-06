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

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml

# --------------------------------------------------------------------------- #
#  3rd‑party imports
# --------------------------------------------------------------------------- #
from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data_temp.dataset import DataConfig

# --------------------------------------------------------------------------- #
#  Training‑level hyper‑parameters
# --------------------------------------------------------------------------- #


class TrainingParams(BaseModel):
    """Hyper‑parameters that control optimisation."""

    train_epochs: int = Field(..., ge=1, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., ge=1, description="Batch size for training")
    optimizer: Literal["adamw", "adam"] = Field("adamw", description="Optimizer to use")
    weight_decay: float = Field(
        0.0, ge=0, description="Weight decay for regularisation"
    )

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
    target_length_seconds: Optional[int] = None
    window_selection: Literal["random", "center"] = "random"

    model_config = ConfigDict(extra="forbid")


class ModelSpec(BaseModel):
    """All parameters required to *instantiate* the network."""

    name: str
    pretrained: bool = True
    device: str = "cuda"
    audio_config: Optional[AudioConfig] = None

    # Fields specifically for CLIP models
    text_model_name: Optional[str] = None
    projection_dim: Optional[int] = None
    temperature: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
#  Top‑level run‑configuration
# --------------------------------------------------------------------------- #


class SchedulerConfig(BaseModel):
    """Configuration for learning rate schedulers."""

    name: Literal["cosine", "linear", "none"] = Field(
        "none", description="Scheduler type to use"
    )
    warmup_steps: int = Field(
        0, ge=0, description="Number of steps to warm up learning rate"
    )
    min_lr: float = Field(
        0.0, ge=0, description="Minimum learning rate for cosine annealing"
    )

    model_config = ConfigDict(extra="forbid")


class RunConfig(BaseModel):
    """Everything needed for a single *training run*."""

    # required
    model_spec: ModelSpec
    training_params: TrainingParams
    dataset_config: str
    output_dir: str

    # optional / misc
    preprocessing: Optional[str] = None
    sr: int = 16000
    logging: Literal["mlflow", "wandb"] = "mlflow"
    label_type: Literal["supervised", "text"] = Field(
        "supervised",
        description=(
            "How to use labels: 'supervised' for classification, "
            "'text' for CLIP training"
        ),
    )

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None

    # Distributed training options
    distributed: bool = Field(
        False,
        description=(
            "Whether to use distributed training (automatically enabled in Slurm)"
        ),
    )
    distributed_backend: Literal["nccl"] = Field(
        "nccl", description="Backend for distributed training (nccl for GPU training)"
    )
    distributed_port: int = Field(
        29500, description="Base port for distributed training communication"
    )

    augmentations: List[Augment] = Field(default_factory=list)
    loss_function: Literal["cross_entropy", "bce", "contrastive", "clip"]

    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    run_name: Optional[str] = None
    wandb_project: str = "audio‑experiments"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # Debug mode
    debug_mode: bool = False

    # ------------------------------
    # custom pre‑processing of augments
    # ------------------------------
    @field_validator("augmentations", mode="before")
    @classmethod
    def _flatten_augments(cls, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert YAML single‑key mapping style into flat dicts.

        YAML allows:
            - noise: {noise_dirs: [...], snr_db_range: [-5, 20], augmentation_prob: 0.5}
        This turns into {kind:"noise", noise_dirs: [...], ...} so the discriminated
        union resolves correctly.

        Returns
        -------
        List[Dict[str, Any]]
            List of flattened augmentation dictionaries
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


def load_config(
    path: str | Path, config_type: Literal["run", "data"] = "run"
) -> RunConfig | DataConfig:
    """Read YAML at *path*, validate, and return a **RunConfig** instance.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file
    config_type : Literal["run", "data"]
        Type of configuration to load

    Returns
    -------
    RunConfig | DataConfig
        Validated configuration object

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    NotImplementedError
        If config_type is not "run" or "data"
    """

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
