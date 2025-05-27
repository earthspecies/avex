from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, Self

import semver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data_temp.config import DatasetConfig
from esp_data_temp.transforms import (
    RegisteredTransformConfigs,
    transform_from_config,
)

from .dataset_utils import GSPath


class DatasetInfo(BaseModel):
    """A Pydantic base model for the info (cfg) of a dataset.

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
    >>> info = DatasetInfo(
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

    license: str = Field(
        default_factory=lambda: "unknown",
        description="License for the dataset, if applicable",
    )

    changelog: str = Field(
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
        if not v:
            raise ValueError("Split paths cannot be empty.")
        for _, value in v.items():
            # Check if the location is a cloud path
            if value.startswith("gs://"):
                path = GSPath(value)
                if not path.exists():
                    raise ValueError(f"Cloud path {value} does not exist.")
            else:
                # Check if the local path exists
                path = Path(value)
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


class Dataset(ABC):
    """Abstract base class defining the interface for ESP datasets.
    Any new dataset should inherit from this class to be added to the registry
    of available ESP datasets.

    Attributes
    ----------
    info : DatasetInfo
        Required attribute containing metadata about the dataset.
        Must be defined by all implementing classes.

    Methods
    -------
    _load(split: Literal["train", "validation"]) -> pd.DataFrame
        Required method to load a specific split of the dataset.
    __len__() -> int
        Required method to return the number of samples in the dataset.
    __iter__() -> Iterator[Dict[str, Any]]
        Required method to iterate over the samples in the dataset.
    __getitem__(idx: int) -> Dict[str, Any]
        Required method to get a specific sample from the dataset.
    """

    info: DatasetInfo = None

    def __init__(self, output_take_and_give: dict[str, str] = None) -> None:
        """A DatasetConfig can be passed to the constructor to, for instance,
        apply transformations to the dataset during instantiation or modify its
        fields of output.

        Parameters
        ----------
        dataset_config : DatasetConfig
            The configuration for the dataset.
        """
        self.output_take_and_give = output_take_and_give

    @property
    def available_splits(self) -> Sequence[str]:
        """Get the available splits of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of split names available in the dataset.
        """
        raise NotImplementedError

    @property
    def columns(self) -> Sequence[str]:
        """Get the columns of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of column names in the dataset.
        """
        raise NotImplementedError

    # TODO (discuss) do we want the underlying data object to be public ?
    # @property
    # @abstractmethod
    # def data(self) -> Sequence[Any]:
    #     """The dataset as a sequence of objects.

    #     Returns
    #     -------
    #     Sequence[Any]
    #         The dataset as a sequence of objects.
    #     """
    #     raise NotImplementedError

    @abstractmethod
    def _load(self, split: str) -> Sequence[Any]:
        """Load one split of the dataset.
        It should apply transformations if any in self.dataset_config.

        Parameters
        ----------
        split : str
            Which split of the dataset to load.

        Returns
        -------
        Sequence[Any]
            The requested split of the dataset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        dataset_config: DatasetConfig,
    ) -> Self:
        """Create a dataset instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetInfo
            The configuration for the dataset.

        Returns
        -------
        Self
            The dataset instance.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Get the iterator over the dataset.

        Returns
        -------
        Iterator[Dict[str, Any]]
            Iterator over samples in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the sample data

        Raises
        ------
        IndexError
            If the index is out of bounds
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the dataset.

        This method should provide a human-readable description of the dataset,
        typically including its name, version, and basic statistics.

        Returns
        -------
        str
            A string representation of the dataset
        """
        raise NotImplementedError

    def apply_transformations(
        self, transformations: list[RegisteredTransformConfigs]
    ) -> list[Any]:
        """Apply the given list of transformations to the dataset.

        This method applies each transformation in sequence to the dataset's data.
        The transformations are applied in-place, modifying the dataset's data.

        Parameters
        ----------
        transformations : list[RegisteredTransformConfigs]
            List of transformation configurations to apply to the dataset.

        Returns
        -------
        list[Any]
            The metadata as a list of objects.

        Raises
        -------
        RuntimeError
            If the dataset's data is not loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() first.")

        metadata_list = []
        for cfg in transformations:
            transform = transform_from_config(cfg)
            self._data, metadata = transform(self._data)
            metadata_list.append(metadata)

            # TODO (milad): what about metadata?
        return metadata_list


# Global registry instance
_dataset_registry: dict[str, Dataset] = {}


def register_dataset(cls: Dataset) -> Dataset:
    """A decorator to register a dataset class.

    Parameters
    ----------
    cls : Type[RegisteredDataset]
        The dataset class to register

    Returns
    -------
    Type[RegisteredDataset]
        The registered dataset class
    """
    name = cls.info.name
    _dataset_registry[name] = cls
    return cls


def list_registered_datasets() -> list[str]:
    """List all registered datasets.

    Returns
    -------
    list[str]
        List of dataset names
    """
    return list(_dataset_registry.keys())


def print_registered_datasets() -> None:
    """Print all registered datasets."""
    for dataset_class in _dataset_registry.values():
        print(dataset_class.info.model_dump_json(indent=2))


def dataset_from_config(dataset_config: DatasetConfig) -> Dataset:
    """Load a dataset from a configuration.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The configuration for the dataset.
    split : str
        The split to load. Can be "train" or "validation".

    Returns
    -------
    Dataset
        The requested dataset instance

    Raises
    ------
    ValueError
        If the dataset is not registered
    """
    _dataset_class = _dataset_registry.get(dataset_config.dataset_name)
    if _dataset_class is None:
        raise ValueError(f"Dataset '{dataset_config.dataset_name}' is not registered.")
    return _dataset_class.from_config(dataset_config)
