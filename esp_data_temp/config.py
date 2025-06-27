from typing import Any

from pydantic import BaseModel, field_validator

from .transforms import RegisteredTransformConfigs


class DatasetConfig(BaseModel):
    dataset_name: str
    transformations: list[RegisteredTransformConfigs] | None = None

    # TODO (milad) Commented out until find a reason to enable
    #              dc_field -> from dataclasses import field as dc_field
    # read_csv_kwargs: Dict[str, Any] = dc_field(default_factory=dict)

    multi_label: bool | None = None
    sample_rate: int | None = None  # Sample rate for audio data
    metrics: list[str] | None = None
    audio_path_col: str | None = None
    output_take_and_give: dict[str, str] | None = None
    split: str = "train"
    strong_detection: bool = False  # TODO: TEMP - where to store this?

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:  # noqa: ANN401
        if v in ("None", "none"):
            return None
        return v
