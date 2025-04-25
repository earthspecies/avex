from dataclasses import field as dc_field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .transformations import TransformCfg


class DataConfig(BaseModel):
    dataset_name: str
    label_column: str
    label_type: Literal["supervised", "self-supervised"]
    transformations: Optional[List[TransformCfg]] = None  # <- changed
    # TODO (milad) what is dc_field? 🤔
    read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)
