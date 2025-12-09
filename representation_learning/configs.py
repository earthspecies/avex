"""
configs.py
~~~~~~~~~~
Canonical **Pydantic v2** data‑classes for training‑run YAML files.

The schema is deliberately strict (`extra='forbid'`) so that typos in
configuration files raise immediately.

Usage
-----
>>> from pathlib import Path
>>> import yaml
>>> from representation_learning.configs import RunConfig
>>> # RunConfig is available for configuration management
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.v1.utils import deep_update
from pydantic_settings import BaseSettings, CliSettingsSource, YamlConfigSettingsSource

if TYPE_CHECKING:
    from representation_learning.data.configs import (
        BenchmarkEvaluationConfig,
        DatasetCollectionConfig,
    )

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Training‑level hyper‑parameters
# --------------------------------------------------------------------------- #


class TrainingParams(BaseModel):
    """Hyper‑parameters that control optimisation."""

    train_epochs: int = Field(..., ge=1, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., ge=1, description="Batch size for training")
    optimizer: Literal["adamw", "adam", "adamw8bit"] = Field("adamw", description="Optimizer to use")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay for regularisation")

    # Optional override for Adam/AdamW beta parameters (β₁, β₂).  If omitted
    # we fall back to the libraries' defaults (0.9, 0.999).
    adam_betas: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Override the (beta1, beta2) coefficients for Adam-type optimisers",
    )

    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    # Frequency (in *iterations*) of logging benchmarking stats & progress
    log_steps: int = Field(100, ge=1, description="Log interval in training steps")

    # Gradient checkpointing for memory optimization
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory")

    # Gradient clipping for training stability
    gradient_clip_val: Optional[float] = Field(
        None,
        ge=0,
        description=("Maximum gradient norm for clipping. If None, no clipping is applied."),
    )

    # Two-stage fine-tuning parameters
    freeze_backbone_epochs: int = Field(
        0,
        ge=0,
        description=(
            "If >0, keep backbone parameters frozen for this many initial epochs "
            "(train only the classification head). At the end of the freeze "
            "period the backbone is unfrozen and optimisation restarts with a "
            "fresh learning-rate schedule."
        ),
    )
    second_stage_lr: Optional[float] = Field(
        None,
        ge=0,
        description=("Learning rate to use after unfreezing the backbone. If omitted we default to current lr × 0.1."),
    )
    second_stage_warmup_steps: Optional[int] = Field(
        None,
        ge=0,
        description=("Warm-up steps for the second stage. Defaults to scheduler.warmup_steps if not provided."),
    )

    # Skip validation during training
    skip_validation: bool = Field(
        False,
        description="Skip validation epochs during training (train-only mode)",
    )

    # Learning rate scheduler parameters
    warmup_epochs: int = Field(
        5,
        ge=0,
        description="Number of warmup epochs for learning rate scheduling",
    )
    scheduler_type: Literal["none", "cosine", "linear", "step"] = Field(
        "cosine",
        description="Type of learning rate scheduler to use",
    )

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
#  Data‑augmentation sections
# --------------------------------------------------------------------------- #


class NoiseAugment(BaseModel):
    """Configuration for noise augmentation during training.

    This augmentation adds background noise to audio samples to improve
    model robustness and generalization.
    """

    kind: Literal["noise"] = "noise"
    noise_dirs: List[str]
    snr_db_range: Tuple[int, int] = Field(..., min_length=2, max_length=2)
    augmentation_prob: float = Field(..., ge=0, le=1)
    mask_signal_prob: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Probability of masking the original signal and using only noise",
    )

    model_config = ConfigDict(extra="forbid")


class MixupAugment(BaseModel):
    """Configuration for mixup augmentation during training.

    This augmentation creates convex combinations of pairs of examples and their labels
    to improve model generalization and robustness.
    """

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
    """Configuration for audio processing parameters.

    This class defines how audio data should be processed, including sample rate,
    windowing parameters, and representation type.
    """

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
    center: bool = True

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "sample_rate",
        "n_fft",
        "hop_length",
        "win_length",
        "n_mels",
        "target_length_seconds",
    )
    @classmethod
    def validate_positive_int(cls, v: Optional[int]) -> Optional[int]:
        """Validate that integer fields are positive.

        Returns
        -------
        Optional[int]
            The validated integer value if positive, or None if the input was None.

        Raises
        ------
        ValueError
            If the value is not None and is less than or equal to 0.
        """
        if v is not None and v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


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

    # Free-form overrides for the EAT backbone (Data2VecMultiConfig).
    eat_cfg: Optional[dict[str, Any]] = None  # noqa: ANN401

    # When true the EAT model is instantiated for self-supervised pre-training.
    pretraining_mode: Optional[bool] = None
    handle_padding: Optional[bool] = None

    # EAT HF specific configuration: TODO handling for model-specific configs
    fairseq_weights_path: Optional[str] = Field(None, description="Path to fairseq checkpoint for EAT HF model")

    # EAT HF audio normalization parameters
    eat_norm_mean: Optional[float] = Field(
        -4.268,
        description="Normalization mean for EAT HF model (default for general audio)",
    )
    eat_norm_std: Optional[float] = Field(
        4.569,
        description="Normalization std for EAT HF model (default for general audio)",
    )

    # EfficientNet variant configuration
    efficientnet_variant: Literal["b0", "b1"] = Field("b0", description="EfficientNet variant to use (b0 or b1)")

    # BEATs-specific configuration
    # TODO: general approach for model-specific configs
    use_naturelm: Optional[bool] = Field(None, description="Whether to use NatureLM for BEATs model")
    fine_tuned: Optional[bool] = Field(None, description="Whether to use fine-tuned weights for BEATs model")

    # BirdNet-specific configuration
    language: Optional[str] = Field(None, description="Language model for BirdNet (e.g., 'en_us', 'en_uk')")

    # EAT HF model ID for HuggingFace model loading
    model_id: Optional[str] = Field(
        "worstchan/EAT-base_epoch30_pretrain",
        description=(
            "HuggingFace model repository ID or local path for EAT HF model "
            "(e.g., 'worstchan/EAT-base_epoch30_pretrain')"
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("eat_cfg")
    @classmethod
    def validate_eat_cfg(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Validate that eat_cfg contains only serializable values.

        Returns
        -------
        Optional[dict[str, Any]]
            The validated configuration dictionary if all values are serializable,
            or None if the input was None.
        """
        if v is None:
            return v

        def is_serializable(obj: object) -> bool:
            """Check if an object is JSON serializable.

            Returns
            -------
            bool
                True if the object is JSON serializable, False otherwise.
            """
            try:
                import json

                json.dumps(obj)
                return True
            except (TypeError, ValueError):
                return False

        def check_dict(d: dict) -> None:
            """Recursively check if all values in a dict are serializable.

            Raises
            ------
            ValueError
                If any value in the dictionary is not JSON serializable.
            """
            for key, value in d.items():
                if isinstance(value, dict):
                    check_dict(value)
                elif not is_serializable(value):
                    raise ValueError(f"Non-serializable value found in eat_cfg: {key}={value}")

        check_dict(v)
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that device is a valid torch device string.

        Returns
        -------
        str
            The validated device string.

        Raises
        ------
        ValueError
            If the device string is not one of the allowed values ('cpu', 'cuda').
        """
        if v not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {v}. Must be one of: cpu, cuda")
        return v


# --------------------------------------------------------------------------- #
#  Probe configuration
# --------------------------------------------------------------------------- #


class ProbeConfig(BaseModel):
    """Configuration for different types of probing strategies.

    This class defines the configuration for various probe types including
    linear probes, MLPs, attention mechanisms, and sequence models.
    """

    probe_type: Literal[
        "linear",
        "mlp",
        "attention",
        "lstm",
        "transformer",
    ] = Field("linear", description="Type of probe to use")

    aggregation: Literal["mean", "max", "cls_token", "none"] = Field(
        "mean", description="How to aggregate multiple dimensional embeddings"
    )

    input_processing: Literal["flatten", "sequence", "pooled", "none"] = Field(
        "pooled",
        description="How to process input embeddings before feeding to probe",
    )

    target_layers: List[str] = Field(..., description="List of layer names to extract embeddings from")

    freeze_backbone: bool = Field(True, description="Whether to freeze the backbone model during probing")

    # MLP-specific parameters
    hidden_dims: Optional[List[int]] = Field(None, description="Hidden dimensions for MLP probe (e.g., [512, 256])")

    dropout_rate: float = Field(0.1, ge=0, le=1, description="Dropout rate for non-linear probes")

    activation: Literal["relu", "gelu", "tanh", "swish"] = Field(
        "relu", description="Activation function for non-linear probes"
    )

    # Attention/Transformer-specific parameters
    num_heads: Optional[int] = Field(
        None,
        ge=1,
        description="Number of attention heads for attention/transformer probes",
    )

    attention_dim: Optional[int] = Field(None, ge=1, description="Dimension for attention mechanism")

    num_layers: Optional[int] = Field(None, ge=1, description="Number of layers for transformer/LSTM probes")

    # LSTM-specific parameters
    lstm_hidden_size: Optional[int] = Field(None, ge=1, description="Hidden size for LSTM probe")

    bidirectional: bool = Field(False, description="Whether to use bidirectional LSTM")

    # Sequence processing parameters
    max_sequence_length: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum sequence length for sequence-based probes",
    )

    use_positional_encoding: bool = Field(
        False,
        description="Whether to add positional encoding for sequence probes",
    )

    # Target length configuration
    target_length: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Target length in samples for audio processing. If None, will be computed from base_model.audio_processor"
        ),
    )

    # Training mode configuration
    online_training: Optional[bool] = Field(
        None,
        description=(
            "Whether to train online (using raw audio) or offline (using "
            "pre-computed embeddings). If None, automatically determined based "
            "on aggregation method: 'mean'/'max' -> offline, 'none'/'cls_token' "
            "-> online. Online training is required for sequence-"
            "based probes (LSTM, attention, transformer)."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_probe_configuration(self) -> "ProbeConfig":
        """Validate the overall probe configuration for consistency.

        Returns
        -------
        ProbeConfig
            The validated probe configuration.

        Raises
        ------
        ValueError
            If the configuration is invalid or inconsistent.
        """
        # Validate MLP-specific parameters
        if self.probe_type in ["mlp"]:
            if self.hidden_dims is None:
                raise ValueError("MLP probe requires hidden_dims to be specified")
            if len(self.hidden_dims) == 0:
                raise ValueError("MLP probe hidden_dims cannot be empty")
            if any(dim <= 0 for dim in self.hidden_dims):
                raise ValueError("MLP probe hidden_dims must all be positive")

        # Validate attention/transformer-specific parameters
        if self.probe_type in [
            "attention",
            "transformer",
        ]:
            if self.num_heads is None:
                raise ValueError(f"{self.probe_type} probe requires num_heads to be specified")
            if self.attention_dim is None:
                raise ValueError(f"{self.probe_type} probe requires attention_dim to be specified")
            if self.num_layers is None:
                raise ValueError(f"{self.probe_type} probe requires num_layers to be specified")

        # Validate LSTM-specific parameters
        if self.probe_type in ["lstm"]:
            if self.lstm_hidden_size is None:
                raise ValueError("LSTM probe requires lstm_hidden_size to be specified")
            if self.num_layers is None:
                raise ValueError("LSTM probe requires num_layers to be specified")

        # Enforce: offline training requires a frozen backbone
        # If online_training is explicitly set to False (offline), the backbone
        # must be frozen to use pre-computed embeddings safely.
        if self.online_training is False and not self.freeze_backbone:
            raise ValueError(
                "When online_training=False, freeze_backbone must be True to use offline embedding-based probing."
            )

        # Enforce: unfrozen backbone requires online training
        # If freeze_backbone is False, online_training must be True for fine-tuning
        if not self.freeze_backbone and self.online_training is False:
            raise ValueError("When freeze_backbone=False, online_training must be True for fine-tuning with raw audio.")

        return self

    def get_training_mode(self) -> bool:
        """Determine whether to use online training based on configuration.

        Returns
        -------
        bool
            True for online training (raw audio), False for offline training "
            "(embeddings)"
        """
        # If explicitly set, use that value
        if self.online_training is not None:
            return self.online_training

        # If backbone is not frozen, we must train online (fine-tuning)
        if not self.freeze_backbone:
            return True

        # Auto-determine based on aggregation method
        if self.aggregation in ["mean", "max", "cls_token"]:
            return False  # Offline training with pre-computed embeddings
        else:
            return True  # Online training with raw audio (required for sequence probes)


# Predefined probe configurations for common use cases
PROBE_CONFIGS = {
    "simple_linear": ProbeConfig(
        probe_type="linear",
        aggregation="mean",
        input_processing="pooled",
        target_layers=["layer_12"],
    ),
    "sequence_lstm": ProbeConfig(
        probe_type="lstm",
        aggregation="none",
        input_processing="sequence",
        target_layers=["layer_8", "layer_12"],
        lstm_hidden_size=256,
        num_layers=2,
        bidirectional=True,
    ),
    "attention_probe": ProbeConfig(
        probe_type="attention",
        aggregation="none",
        input_processing="sequence",
        target_layers=["layer_6", "layer_10"],
        num_heads=8,
        attention_dim=512,
        num_layers=2,
    ),
    "mlp_probe": ProbeConfig(
        probe_type="mlp",
        aggregation="mean",
        input_processing="pooled",
        target_layers=["layer_12"],
        hidden_dims=[512, 256],
        dropout_rate=0.2,
        activation="gelu",
    ),
    "transformer_probe": ProbeConfig(
        probe_type="transformer",
        aggregation="none",
        input_processing="sequence",
        target_layers=["layer_6", "layer_8", "layer_10", "layer_12"],
        num_heads=8,
        attention_dim=512,
        num_layers=3,
        use_positional_encoding=True,
    ),
}


# --------------------------------------------------------------------------- #
#  Top-level run-configuration
# --------------------------------------------------------------------------- #


class SchedulerConfig(BaseModel):
    """Configuration for learning rate schedulers."""

    name: Literal["cosine", "linear", "none"] = Field("none", description="Scheduler type to use")
    warmup_steps: int = Field(0, ge=0, description="Number of steps to warm up learning rate")
    min_lr: float = Field(0.0, ge=0, description="Minimum learning rate for cosine annealing")

    model_config = ConfigDict(extra="forbid")


class BaseCLIConfig(BaseSettings):
    """
    A base class for configs that can be loaded from a YAML file and CLI arguments.
    """

    @classmethod
    def from_sources(cls, yaml_file: str | Path, cli_args: tuple[str, ...]) -> "RunConfig":
        """
        Create a RunConfig object from a YAML file and CLI arguments. If there are any
        conflicts, the CLI arguments will take precedence over the YAML file.

        Parameters
        ----------
        yaml_file : str | Path
            Path to the YAML configuration file
        cli_args : tuple[str, ...]
            Tuple of CLI arguments to override the YAML file

        Returns
        -------
        RunConfig
            Validated configuration object

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist
        """

        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file {yaml_file} does not exist")

        yaml_values = YamlConfigSettingsSource(cls, yaml_file=yaml_file)
        cli_values = CliSettingsSource(cls, cli_parse_args=["--" + opt for opt in cli_args])
        final_values = deep_update(yaml_values(), cli_values())
        return cls.model_validate(final_values)


class ClusteringEvalConfig(BaseModel):
    """Configuration for clustering evaluation during training."""

    enabled: bool = Field(False, description="Enable clustering evaluation")
    frequency: int = Field(5, ge=1, description="Evaluate clustering every N epochs")
    layers: str = Field(
        "last_layer",
        description="Comma-separated layer names for embedding extraction",
    )
    use_validation_set: bool = Field(
        True,
        description="Use validation set for clustering (else use train set)",
    )
    max_samples: Optional[int] = Field(None, ge=100, description="Maximum samples to use (None = use all)")
    run_before_training: bool = Field(False, description="Run clustering evaluation before the first epoch")

    model_config = ConfigDict(extra="forbid")


class RunConfig(BaseCLIConfig, extra="forbid", validate_assignment=True):
    """Everything needed for a single *training run*."""

    # --------------------------------------------------------------------------- #
    #  Clustering evaluation configuration
    # --------------------------------------------------------------------------- #
    """Everything needed for a single *training run*."""

    # required
    model_spec: ModelSpec
    training_params: TrainingParams
    dataset_config: DatasetCollectionConfig
    output_dir: str

    # optional / misc
    preprocessing: Optional[str] = None
    sr: int = 16000
    logging: Literal["mlflow", "wandb", "none"] = "mlflow"
    # TODO : make default uri localhost ?
    logging_uri: str = "http://127.0.0.1:5000/"
    label_type: Literal["supervised", "text", "self_supervised"] = Field(
        "supervised",
        description=(
            "How to use labels: 'supervised' for classification, "
            "'text' for CLIP training, 'self_supervised' for self-supervised learning"
        ),
    )

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None

    # Distributed training options
    distributed: bool = Field(
        False,
        description=("Whether to use distributed training (automatically enabled in Slurm)"),
    )
    distributed_backend: Literal["nccl"] = Field(
        "nccl",
        description="Backend for distributed training (nccl for GPU training)",
    )
    distributed_port: int = Field(29500, description="Base port for distributed training communication")

    augmentations: List[Augment] = Field(default_factory=list)
    # Allow common aliases for BCE to avoid validation errors in older configs
    loss_function: Literal[
        "cross_entropy",
        "bce",
        "binary_cross_entropy",
        "contrastive",
        "clip",
        "focal",
    ]

    # Enable multi-label classification
    multilabel: bool = Field(False, description="Whether to use multi-label classification")

    # Metrics to compute during training
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy"],
        description="List of metrics to compute during training",
    )

    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    run_name: Optional[str] = None
    wandb_project: str = "audio‑experiments"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    # Clustering evaluation configuration
    clustering_eval: Optional[ClusteringEvalConfig] = Field(
        None,
        description="Configuration for clustering evaluation during training",
    )

    # Debug mode
    debug_mode: bool = False

    @field_validator("dataset_config", mode="before")
    @classmethod
    def _maybe_read_from_yml(cls, raw: Any) -> Any:  # noqa: ANN401
        # If it's a string, treat it as a path to a YAML file
        if isinstance(raw, str) and raw.endswith((".yml", ".yaml")):
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"Dataset config file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Otherwise, let normal validation proceed
        return raw

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

        # Check if multilabel is True but loss function isn't BCE/Focal
        if data.get("multilabel", False) and v not in {"bce", "focal"}:
            raise ValueError(f"When multilabel=True, loss_function must be 'bce' or 'focal' (got '{v}' instead)")

        # For self-supervised runs we don't impose any loss-type restrictions
        if data.get("label_type") == "self_supervised":
            return v

        # Check if loss is clip/contrastive but label_type isn't text
        if v in ("clip", "contrastive") and data.get("label_type") != "text":
            raise ValueError(f"Loss function '{v}' requires label_type='text' (got '{data.get('label_type')}' instead)")

        return v


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment in evaluation."""

    run_name: str = Field(..., description="Name of the experiment run")
    run_config: RunConfig
    pretrained: bool = Field(True, description="Whether to use pretrained weights")

    # Legacy layers field (deprecated, use probe_config instead)
    layers: Optional[str] = Field(
        None,
        description="Deprecated: Use probe_config.target_layers instead. "
        "Comma separated list of layer names to extract embeddings from. "
        "Use probe_config.target_layers instead.",
    )

    # New flexible probe configuration
    probe_config: Optional[ProbeConfig] = Field(
        None,
        description="Configuration for the probing strategy. If not provided, "
        "uses legacy linear probe with 'layers' field.",
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

    # Whether to freeze the backbone and train only the linear probe
    # This is now controlled by probe_config.freeze_backbone if probe_config is provided
    frozen: Optional[bool] = Field(
        None,
        description="Deprecated: Use probe_config.freeze_backbone instead. "
        "Whether to update base model weights during linear probing. "
        "Use probe_config.freeze_backbone instead.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("run_config", mode="before")
    @classmethod
    def _maybe_read_from_yml(cls, raw: Any) -> Any:  # noqa: ANN401
        # If it's a string, treat it as a path to a YAML file
        if isinstance(raw, str) and raw.endswith((".yml", ".yaml")):
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"Dataset config file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Otherwise, let normal validation proceed
        return raw

    @model_validator(mode="after")
    def validate_experiment_config(self) -> "ExperimentConfig":
        """Validate experiment configuration and handle legacy field migration.

        Returns
        -------
        ExperimentConfig
            The validated experiment configuration.

        Raises
        ------
        ValueError
            If the configuration is invalid or inconsistent.
        """
        # Handle legacy configuration migration
        if self.probe_config is None:
            # Legacy mode: create a default linear probe config from legacy fields
            if self.layers is None:
                raise ValueError(
                    "Either probe_config or layers must be provided. Use probe_config for flexible probing strategies."
                )

            # Create a default linear probe configuration
            default_config = ProbeConfig(
                probe_type="linear",
                aggregation="mean",
                input_processing="pooled",
                target_layers=self.layers.split(","),
                freeze_backbone=(self.frozen if self.frozen is not None else True),
            )

            self.probe_config = default_config

            # Log deprecation warning
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Experiment '{self.run_name}' uses deprecated 'layers' and "
                f"'frozen' fields. "
                f"Consider migrating to probe_config for more flexibility."
            )
        else:
            # New mode: validate that legacy fields are not conflicting
            if self.layers is not None:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Experiment '{self.run_name}' has both probe_config and "
                    f"legacy 'layers' field. "
                    f"Ignoring 'layers' field in favor of "
                    f"probe_config.target_layers."
                )

            if self.frozen is not None:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Experiment '{self.run_name}' has both probe_config and "
                    f"legacy 'frozen' field. "
                    f"Ignoring 'frozen' field in favor of "
                    f"probe_config.freeze_backbone."
                )

        return self

    def get_target_layers(self) -> List[str]:
        """Get the target layers for this experiment.

        Returns
        -------
        List[str]
            List of target layer names

        Raises
        ------
        ValueError
            If no target layers are configured
        """
        if self.probe_config is not None:
            return self.probe_config.target_layers
        elif self.layers is not None:
            return self.layers.split(",")
        else:
            raise ValueError("No target layers specified in experiment configuration")

    def is_frozen(self) -> bool:
        """Check if the backbone should be frozen for this experiment.

        Returns
        -------
        bool
            True if the backbone should be frozen, False otherwise
        """
        if self.probe_config is not None:
            return self.probe_config.freeze_backbone
        return True  # Default to frozen for backward compatibility

    def get_probe_type(self) -> str:
        """Get the probe type for this experiment.

        Returns
        -------
        str
            The probe type (e.g., 'linear', 'mlp', 'attention', 'lstm', 'transformer')
        """
        if self.probe_config is not None:
            return self.probe_config.probe_type
        else:
            return "linear"  # Default to linear for backward compatibility

    def get_probe_specific_params(self) -> Dict[str, Any]:
        """Get probe-specific parameters for this experiment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of probe-specific parameters
        """
        if self.probe_config is not None:
            # Return all probe-specific parameters as a dictionary
            params = {}

            # MLP parameters
            if self.probe_config.hidden_dims is not None:
                params["hidden_dims"] = self.probe_config.hidden_dims
            if self.probe_config.dropout_rate is not None:
                params["dropout_rate"] = self.probe_config.dropout_rate
            if self.probe_config.activation is not None:
                params["activation"] = self.probe_config.activation

            # Attention/Transformer parameters
            if self.probe_config.num_heads is not None:
                params["num_heads"] = self.probe_config.num_heads
            if self.probe_config.attention_dim is not None:
                params["attention_dim"] = self.probe_config.attention_dim
            if self.probe_config.num_layers is not None:
                params["num_layers"] = self.probe_config.num_layers

            # LSTM parameters
            if self.probe_config.lstm_hidden_size is not None:
                params["lstm_hidden_size"] = self.probe_config.lstm_hidden_size
            if self.probe_config.bidirectional is not None:
                params["bidirectional"] = self.probe_config.bidirectional

            # Sequence processing parameters
            if self.probe_config.max_sequence_length is not None:
                params["max_sequence_length"] = self.probe_config.max_sequence_length
            if self.probe_config.use_positional_encoding is not None:
                params["use_positional_encoding"] = self.probe_config.use_positional_encoding

            return params
        else:
            return {}  # No probe-specific parameters for legacy configuration

    def get_aggregation_method(self) -> str:
        """Get the aggregation method for this experiment.

        Returns
        -------
        str
            The aggregation method (e.g., 'mean', 'max', 'cls_token', 'none')
        """
        if self.probe_config is not None:
            return self.probe_config.aggregation
        else:
            return "mean"  # Default to mean for backward compatibility

    def get_input_processing_method(self) -> str:
        """Get the input processing method for this experiment.

        Returns
        -------
        str
            The input processing method (e.g., 'flatten', 'sequence', 'pooled', 'none')
        """
        if self.probe_config is not None:
            return self.probe_config.input_processing
        else:
            return "pooled"  # Default to pooled for backward compatibility

    def get_training_mode(self) -> bool:
        """Get the training mode for this experiment.

        Returns
        -------
        bool
            True for online training (raw audio), False for offline training "
            "(embeddings)"
        """
        if self.probe_config is not None:
            return self.probe_config.get_training_mode()
        else:
            # Legacy mode: determine based on frozen status
            # If not frozen, we're fine-tuning (online)
            # If frozen, we're doing linear probing (offline)
            return not self.is_frozen()


class EvaluateConfig(BaseCLIConfig, extra="forbid"):
    """Configuration for running evaluation experiments."""

    experiments: List[ExperimentConfig] = Field(..., description="List of experiments to run")
    dataset_config: BenchmarkEvaluationConfig
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

    # Which evaluation phases to run
    eval_modes: List[Literal["probe", "retrieval", "clustering"]] = Field(
        default_factory=lambda: ["probe"],
        description="Configuration for flexible probing system in "
        "run_evaluate.py (probe covers all probe types: linear, MLP, LSTM, "
        "attention, transformer)",
    )

    # Offline embeddings behavior (loading/saving/extraction)
    class OfflineEmbeddingsConfig(BaseModel):
        """Configuration for offline embeddings behavior.

        This class controls how embeddings are loaded, saved, and extracted
        when using offline embedding-based probing.
        """

        memory_limit_gb: int = Field(
            32,
            ge=1,
            description=(
                "Maximum file size (in GB) to load embeddings fully into memory. "
                "Larger files will use HDF5-backed loading."
            ),
        )
        overwrite_embeddings: bool = Field(
            False,
            description=(
                "If False and cached embeddings are found on disk, they will be "
                "loaded instead of recomputed. If True, embeddings are always "
                "recomputed and the cache is overwritten."
            ),
        )
        use_streaming_embeddings: bool = Field(
            False,
            description=(
                "If True, use streaming approach for embedding extraction to prevent "
                "OOM issues. Saves embeddings directly to disk in chunks."
            ),
        )
        cache_size_limit_gb: float = Field(
            8.0,
            ge=1,
            description=(
                "Maximum RAM (in GB) used to cache HDF5 embeddings in memory when "
                "loading for offline probing. If the full dataset doesn't fit, we "
                "cache labels and as many layers as possible within this budget."
            ),
        )
        streaming_chunk_size: int = Field(
            1000,
            ge=100,
            description=("Samples per chunk when using streaming extraction."),
        )
        hdf5_compression: str = Field(
            "gzip",
            description=("HDF5 compression algorithm for embedding storage."),
        )
        hdf5_compression_level: int = Field(
            4,
            ge=1,
            le=9,
            description=("Compression level for gzip (1-9)."),
        )
        auto_chunk_size: bool = Field(
            True,
            description=("Auto-calculate optimal chunk size based on available GPU memory."),
        )
        max_chunk_size: int = Field(
            2000,
            ge=100,
            description=("Maximum chunk size when auto-calculating."),
        )
        min_chunk_size: int = Field(
            100,
            ge=10,
            description=("Minimum chunk size when auto-calculating."),
        )
        batch_chunk_size: int = Field(
            10,
            ge=1,
            description=("Number of batches to process before writing during streaming."),
        )

        model_config = ConfigDict(extra="forbid")

    offline_embeddings: OfflineEmbeddingsConfig = Field(
        default_factory=OfflineEmbeddingsConfig,
        description="Configuration for offline embeddings load/save/extraction",
    )

    # Optional path to append all results to a single CSV file
    results_csv_path: Optional[str] = Field(
        None,
        description=(
            "Optional path to a CSV file where all evaluation results will be "
            "appended. If provided, results from each experiment will be written "
            "to this file with appropriate metadata for tracking across multiple runs."
        ),
    )

    # Control tqdm progress bar verbosity
    disable_tqdm: bool = Field(
        False,
        description=(
            "If True, disable tqdm progress bars during fine-tuning and evaluation. "
            "Useful for reducing output verbosity in automated runs or when logging "
            "to files."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_config", mode="before")
    @classmethod
    def _maybe_read_from_yml(cls, raw: Any) -> Any:  # noqa: ANN401
        # If it's a string, treat it as a path to a YAML file
        if isinstance(raw, str) and raw.endswith((".yml", ".yaml")):
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"Dataset config file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Otherwise, let normal validation proceed
        return raw


# Import dataset configs from data module (dev-only, requires esp-data).
# For API-only installs (no esp-data), these imports are skipped so that the
# public API can be used without the private esp-data dependency. Development
# workflows that rely on these configs must install the dev extras.
try:
    from representation_learning.data.configs import (
        BenchmarkEvaluationConfig,
        DatasetCollectionConfig,
    )
except ImportError:  # pragma: no cover - exercised only in API-only envs
    logger.info(
        "esp_data is not installed; dataset-related configs "
        "(DatasetCollectionConfig, BenchmarkEvaluationConfig) are unavailable. "
        "This is expected for API-only installations. Install the dev extras "
        "to enable training/evaluation configs."
    )
