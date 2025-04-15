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
>>> cfg      = RunConfig(**cfg_dict)   # validated & parsed
"""

from __future__ import annotations

from typing import List, Tuple, Union, Literal, Optional, Any, Dict
from pathlib import Path
import yaml

from pydantic import BaseModel, Field, validator


class TrainingParams(BaseModel):
    train_epochs: int         = Field(..., ge=1)
    lr:           float       = Field(..., gt=0)
    batch_size:   int         = Field(..., ge=1)
    optimizer:    Literal["adamw", "adam"] = "adamw"
    weight_decay: float       = Field(0.0, ge=0)

class NoiseAugment(BaseModel):
    kind:              Literal["noise"] = Field("noise", const=True)
    noise_dirs:        List[str]
    snr_db_range:      Tuple[int, int] = Field(..., min_items=2, max_items=2)
    augmentation_prob: float           = Field(..., ge=0, le=1)

class MixupAugment(BaseModel):
    kind:              Literal["mixup"] = Field("mixup", const=True)
    alpha:             float            = Field(..., gt=0)
    augmentation_prob: float            = Field(..., ge=0, le=1)

# Union of the augmentation types
Augment = Union[NoiseAugment, MixupAugment]


# --------------------------------------------------------------------------- #
#  Top‑level run configuration
# --------------------------------------------------------------------------- #

class RunConfig(BaseModel):
    # Core
    model_name:      str
    dataset_config:  str
    preprocessing:   Optional[str] = None
    sr:              int           = 16000
    logging:         Literal["mlflow", "wandb"] = "mlflow"
    training_params: TrainingParams
    augmentations:   List[Augment]
    loss_function:   Literal["cross_entropy",
                             "bce"
                            ]

    # -------------------------- custom parsing ----------------------------- #
    @validator("augmentations", pre=True)
    def _parse_augments(
        cls, raw_list: List[Any]
    ) -> List[Dict[str, Any]]:                           # noqa: N805
        """
        Convert the YAML representation (each item is a *single‑key* dict)
        into proper Augment objects.
        """
        parsed: List[Augment] = []

        for item in raw_list:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(
                    "Each augmentation entry must be a one‑key mapping, "
                    "e.g.  - mixup: {alpha: 0.4, augmentation_prob: 0.3}"
                )

            aug_type, params = next(iter(item.items()))
            params = params or {}                         # allow e.g. `- mixup`

            # Dispatch to the correct dataclass
            if aug_type == "noise":
                parsed.append(NoiseAugment(**params))
            elif aug_type == "mixup":
                parsed.append(MixupAugment(**params))
            else:
                raise ValueError(f"Unknown augmentation kind '{aug_type}'")

        return parsed


def load_config(path: str | Path) -> RunConfig:
    """Read YAML at *path*, validate, and return a plain dict for convenience."""

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    try:
        cfg = RunConfig.parse_obj(raw)
    except Exception as e:  # noqa: BLE001
        print("Config validation failed:", e)
        raise

    # Return *analysis* section as a dict (run_analysis expects sub‑scriptable)
    return cfg
