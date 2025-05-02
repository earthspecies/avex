from typing import Any

from pydantic import BaseModel, field_validator

from .transformations import RegisteredTransforms


class DatasetConfig(BaseModel):
    dataset_name: str
    transformations: list[RegisteredTransforms] | None = None  # <- changed

    # TODO (milad) Commented out until find a reason to enable
    #              dc_field -> from dataclasses import field as dc_field
    # read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:
        if v in ("None", "none"):
            return None
        return v
