from dataclasses import field as dc_field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .transformations import TransformCfg


class DataConfig(BaseModel):
    dataset_name: str

    # TODO (milad) do we need these?
    # label_column: str
    # label_type: Literal["supervised", "self-supervised"]

    transformations: Optional[List[TransformCfg]] = None  # <- changed
    label_column: str
    label_type: str
    # TODO (milad) what is dc_field? 🤔
    read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)
    multi_label: Optional[bool] = None
    sample_rate: Optional[int] = None  # Sample rate for audio data
