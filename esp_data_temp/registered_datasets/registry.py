import pathlib
from typing import Optional, Protocol, TypeVar

import semver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data_temp.dataset import GSPath


class DatasetInfo(BaseModel):
    """A Pydantic base model for a registered ESP dataset. All datasets
    should subclass this.

    Arguments
    ---------
    name : str
        Name of the dataset
    owner : str | list[str]
        ESP team owner(s) of the dataset
    split_paths : dict[str, str]
        Paths to the dataset splits. The keys are the split names
        and the values are the paths to the splits. The paths can be
    version : str
        Version of the dataset, root dataset is 0.0
    description : str
        Description of the dataset, could act as a README, preferably in markdown format
    sources : list[str] | str
        Source(s) of the dataset e.g. 'Xeno-canto' or a url to website(s),
        or multiple sources in a comma-separated list
    license : Optional[str]
        License for the dataset, if applicable
    changelog : Optional[str]
        Changelog from previous version
    **kwargs : Any (optional)
        Not validated, but can be used to pass additional information

    Examples
    --------
    >>> data = DatasetInfo(
    ...     name="animalspeak",
    ...     owner="marius; masato",
    ...     split_paths={
    ...         "train": "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv",
    ...         "validation": "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv",
    ...     },
    ...     version="0.1.0",
    ...     description="AnimalSpeak dataset",
    ...     sources=["Xeno-canto", "iNaturalist", "Watkins"],
    ...     license="unknown",
    ...     changelog="Initial version",
    ... )
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="allow",
    )

    # required params
    name: str = Field(min_length=1, description="Name of the dataset")

    owner: str = Field(min_length=1, description="ESP team owner(s) of the dataset")

    split_paths: dict = Field(
        description="""Paths to the dataset splits. The keys are the split names
        and the values are the paths to the splits""",
    )

    version: str = Field(min_length=5, description="Version of the dataset")

    description: str = Field(
        min_length=1,
        description="""Description of the dataset, could act as a README,
        preferably in markdown format, and include changelog to previous version""",
    )

    sources: list[str] | str = Field(
        min_length=1,
        description="""Source(s) of the dataset e.g. 'Xeno-canto' or a url to
        website(s) or multiple sources in a comma-separated list""",
    )

    license: Optional[str] = Field(
        default_factory=lambda: "unknown",
        description="License for the dataset, if applicable",
    )

    changelog: Optional[str] = Field(
        default_factory=lambda: "", description="Changelog from previous version"
    )

    @field_validator("split_paths", mode="after")
    @classmethod
    def validate_split_exists(cls, v: dict) -> str:
        """Validate that the split path exists in cloud storage or locally

        Arguments
        ---------
        v : dict[str, str]
            The locations to validate

        Returns
        -------
        dict[str, str]
            The validated locations

        Raises
        ------
        ValueError
            If the location does not exist in cloud storage or locally
        ValueError
            If the location is a directory and is empty
        """
        for _, value in v.items():
            # Check if the location is a cloud path
            if value.startswith("gs://"):
                path = GSPath(value)
                if not path.exists():
                    raise ValueError(f"Cloud path {value} does not exist.")
            else:
                # Check if the local path exists
                path = pathlib.Path(value)
                if not path.exists():
                    raise ValueError(f"Local path {value} does not exist.")

            # if location is directory, check that it is not empty
            if path.is_dir() and not any(path.iterdir()):
                raise ValueError(f"Directory {value} is empty.")

        return v

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validates that the version follows semantic versioning (MAJOR.MINOR.PATCH)
        using the semver package.

        Arguments
        ---------
        v : str
            The version string to validate

        Returns
        -------
        str
            The validated version string

        Raises
        ------
        ValueError
            If the version does not follow semantic versioning
        """
        try:
            semver.VersionInfo.parse(v)
        except ValueError as e:
            raise ValueError(f"""Version '{v}' does not follow semantic versioning
                            (MAJOR.MINOR.PATCH).
                    Error: {str(e)}. See https://semver.org/ for details.""") from e
        return v


class RegisteredDatasetProtocol(Protocol):
    """Protocol for registered dataset classes."""

    info: DatasetInfo


# Type variable for registered dataset classes
RegisteredDataset = TypeVar("RegisteredDataset", bound=RegisteredDatasetProtocol)


class DatasetRegistry:
    """A registry for all registered datasets. This is a singleton class"""

    datasets: dict[str, RegisteredDataset] = {}

    def list(self) -> list[str]:
        """List all registered datasets

        Returns
        -------
        list[str]
            List of dataset names
        """
        return list(self.datasets.keys())

    def print(self) -> None:
        """Print all registered datasets"""
        for dataset in self.datasets.values():
            print(dataset.info.model_dump_json(indent=2))


registry = DatasetRegistry()


def register_dataset(cls: RegisteredDataset) -> RegisteredDataset:
    """A decorator to register a dataset class.

    Arguments
    ---------
    cls : The dataset class to register

    Returns
    -------
    cls : The registered dataset class
    """
    registry.datasets[cls.info.name] = cls
    return cls
