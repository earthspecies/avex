from typing import Any

from pydantic import BaseModel, field_validator

from .transformations import RegisteredTransformConfigs


class DatasetConfig(BaseModel):
    dataset_name: str
    transformations: list[RegisteredTransformConfigs] | None = None

    # TODO (milad) Commented out until find a reason to enable
    #              dc_field -> from dataclasses import field as dc_field
    # read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)

    multi_label: bool | None = None
    sample_rate: int | None = None  # Sample rate for audio data
    metrics: list[str] | None = None

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:
        if v in ("None", "none"):
            return None
        return v
