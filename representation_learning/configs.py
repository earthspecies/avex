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

from esp_data_temp.config import DatasetConfig

# --------------------------------------------------------------------------- #
#  Training‑level hyper‑parameters
# --------------------------------------------------------------------------- #


class TrainingParams(BaseModel):
    """Hyper‑parameters that control optimisation."""

    train_epochs: int = Field(..., ge=1, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., ge=1, description="Batch size for training")
    optimizer: Literal["adamw", "adam", "adamw8bit"] = Field(
        "adamw", description="Optimizer to use"
    )
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
    n_mixup: int = Field(1, ge=1, description="Number of mixup pairs per batch")
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
    # Allow common aliases for BCE to avoid validation errors in older configs
    loss_function: Literal[
        "cross_entropy",
        "bce",
        "binary_cross_entropy",
        "contrastive",
        "clip",
    ]

    # Enable multi-label classification
    multilabel: bool = Field(
        False, description="Whether to use multi-label classification"
    )

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

        But we need:
            - kind: noise
              noise_dirs: [...]
              snr_db_range: [-5, 20]
              augmentation_prob: 0.5

        This transformer ensures both styles work, which means the Pydantic
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

    # ------------------------------
    # validator for multilabel and loss_function
    # ------------------------------
    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, v: str, info: Any) -> str:  # noqa: ANN401
        """Ensure the chosen loss is compatible with other config fields.

        Returns
        -------
        str
            The validated loss-function name.

        Raises
        ------
        ValueError
            If `multilabel` is ``True`` but the loss is not *BCE*, or when a
            contrastive/CLIP loss is requested for a non-text label type.
        """
        data = info.data

        # Map alias to canonical form first
        if v == "binary_cross_entropy":
            v = "bce"

        # Check if multilabel is True but loss function isn't BCE
        if data.get("multilabel", False) and v != "bce":
            raise ValueError(
                f"When multilabel=True, loss_function must be 'bce' (got '{v}' instead)"
            )

        # Check if loss is clip/contrastive but label_type isn't text
        if v in ("clip", "contrastive") and data.get("label_type") != "text":
            raise ValueError(
                f"Loss function '{v}' requires label_type='text' "
                f"(got '{data.get('label_type')}' instead)"
            )

        return v

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment in evaluation."""

    run_name: str = Field(..., description="Name of the experiment run")
    run_config: str = Field(..., description="Path to the run config YAML file")
    pretrained: bool = Field(True, description="Whether to use pretrained weights")
    freeze: bool = Field(True, description="Whether to freeze the backbone")
    layers: str = Field(
        ...,
        description="List of layer names to extract embeddings from, comma separated",
    )

    # Optional path to a trained model checkpoint (ignored when `pretrained=True`)
    checkpoint_path: Optional[str] = Field(
        None,
        description=(
            "Path to the model checkpoint to load when `pretrained` is false. "
            "If not provided, defaults to 'checkpoints/best.pt' relative to the "
            "current working directory."
        ),
    )
    checkpoint_config_path: Optional[str] = Field(
        None,
        description=(
            "Path to the model checkpoint configuration to load. "
            "This is only used for Aves pretrained model. "
        ),
    )

    model_config = ConfigDict(extra="forbid")


class EvaluateConfig(BaseModel):
    """Configuration for running evaluation experiments."""

    experiments: List[ExperimentConfig] = Field(
        ..., description="List of experiments to run"
    )
    dataset_config: str = Field(..., description="Path to the dataset config YAML file")
    save_dir: str = Field(..., description="Directory to save evaluation results")

    # Fine-tuning parameters for linear probing
    training_params: TrainingParams = Field(
        default_factory=lambda: TrainingParams(
            train_epochs=10,
            lr=0.0001,
            batch_size=2,
            optimizer="adamw",
            weight_decay=0.01,
            amp=False,
            amp_dtype="bf16",
        ),
        description="Training parameters for fine-tuning during evaluation",
    )

    device: str = Field(..., description="Device to run the evaluation on")
    seed: int = Field(..., description="Random seed for reproducibility")
    num_workers: int = Field(..., description="Number of workers for evaluation")

    # Whether to freeze the backbone and train only the linear probe
    frozen: bool = Field(
        True,
        description="If True, do not update base model weights during linear probing.",
    )

    model_config = ConfigDict(extra="forbid")


class BenchmarkConfig(BaseModel):
    """Configuration for the entire benchmark suite containing multiple datasets."""

    data_path: str = Field(..., description="Base path for all benchmark datasets")
    datasets: List[DatasetConfig] = Field(
        ..., description="List of benchmark datasets to evaluate"
    )

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
#  Convenience loader
# --------------------------------------------------------------------------- #
def load_config(
    path: str | Path,
    config_type: Literal["run", "data", "evaluate", "benchmark"] = "run",
) -> RunConfig | DatasetConfig | EvaluateConfig | BenchmarkConfig:
    """Read YAML at *path*, validate, and return a **RunConfig** instance.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file
    config_type : Literal["run", "data", "evaluate", "benchmark"]
        Type of configuration to load

    Returns
    -------
    RunConfig | DatasetConfig | EvaluateConfig | BenchmarkConfig
        Validated configuration object

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    NotImplementedError
        If *config_type* is unrecognised
    """

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if config_type == "run":
        return RunConfig.model_validate(raw)
    elif config_type == "data":
        return DatasetConfig.model_validate(raw)
    elif config_type == "evaluate":
        return EvaluateConfig.model_validate(raw)
    elif config_type == "benchmark":
        return BenchmarkConfig.model_validate(raw)
    else:
        raise NotImplementedError("Can only load from run config or data config.")
