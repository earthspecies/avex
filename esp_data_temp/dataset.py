import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
)

import cloudpathlib
import librosa
import numpy as np
import pandas as pd
import semver
import soundfile as sf
from google.cloud.storage.client import Client
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import DatasetConfig
from .transforms import transform_from_config, RegisteredTransformConfigs

# Type variable for registered dataset classes
RegisteredDataset = TypeVar("RegisteredDataset", bound="Dataset")

# Global registry instance
_dataset_registry: dict[str, Type[RegisteredDataset]] = {}


def register_dataset(cls: Type[RegisteredDataset]) -> Type[RegisteredDataset]:
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
    # Create a temporary instance to get the info
    temp_instance = cls()
    name = temp_instance.info.name
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
        # Create temporary instance to access info
        temp_instance = dataset_class()
        print(temp_instance.info.model_dump_json(indent=2))


ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv"
BATS_PATH = "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.train.csv"
BATS_PATH_VALID = (
    "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.valid.csv"
)


@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:
    return cloudpathlib.GSClient(storage_client=Client(), file_cache_mode="close_file")


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the cloudpathlib GSPath that provides a default client.
    This avoids issues when the GOOGLE_APPLICATION_CREDENTIALS variable is not set.
    """

    def __init__(
        self,
        client_path: str | Self | cloudpathlib.AnyPath,
        client: cloudpathlib.GSClient = _get_client(),
    ) -> None:
        super().__init__(client_path, client=client)


class AudioDataset:
    """
    Reads metadata from a CSV, loads audio, and yields a sample dict.

    Expected columns in the CSV:
    * 'filepath'  : str - path to the audio file on disk or a gs:// path.
    * <label_col> : str - value used for the target (e.g. species name).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_config: DatasetConfig,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        metadata: dict | None = None,
        postprocessors: Optional[
            List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ) -> None:
        super().__init__()

        # TODO (milad) transform arg here?

        self.df = df.reset_index(drop=True)
        self.data_config = data_config
        self.preprocessor = preprocessor

        self.audio_path_col = "gs_path"  # modify if your CSV uses a different name

        self.metadata = metadata

        self.postprocessors = postprocessors or []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO
        pass

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator:
        # TODO (milad) do this properly
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        # TODO
        return "TODO"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        if path_str.startswith("gs://"):
            if GSPath is None:
                raise ImportError("cloudpathlib is required to handle gs:// paths.")
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        # Open the audio file. Using the .open('rb') method works for both local and
        # GSPath objects.
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        target_sr = self.data_config.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
            sr = target_sr

        item = {
            "raw_wav": audio.astype(np.float32),
            "text_label": row["label_feature"],
            "label": row.label,
            "path": str(audio_path),
        }

        for proc in self.postprocessors:
            item = proc(item)

        return item


def _get_dataset_from_name(
    name: str,
    split: str = "train",
) -> pd.DataFrame:
    name = name.lower().strip()

    if name == "animalspeak":
        if split == "test":
            return None
        anaimspeak_path = (
            ANIMALSPEAK_PATH_EVAL if split == "valid" else ANIMALSPEAK_PATH
        )
        if ANIMALSPEAK_PATH.startswith("gs://"):
            csv_path = GSPath(anaimspeak_path)
        else:
            csv_path = Path(anaimspeak_path)

        # Read CSV content
        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(
            # lambda x: "gs://" + x
            lambda x: "/home/milad_earthspecies_org/data-migration/marius-highmem/mnt/foundation-model-data/audio_16k/"
            + x
        )  # AnimalSpeak missing gs path
        return df
    elif name == "bats":
        csv_file = (
            BATS_PATH_TEST
            if split == "test"
            else BATS_PATH_VALID
            if split == "valid"
            else BATS_PATH
        )
        # TODO: don't use os.path!
        base_path = os.path.dirname(csv_file).split("egyptian_fruit_bats")[0]
        if csv_file.startswith("gs://"):
            csv_path = GSPath(csv_file)
        else:
            csv_path = Path(csv_file)

        # Read CSV content
        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["path"].apply(
            lambda x: base_path
            + "egyptian_fruit_bats"
            + x.split("egyptian_fruit_bats")[1]
        )  # bats missing gs path
        return df
    else:
        raise NotImplementedError("Dataset not supported")


def get_dataset_dummy(
    data_config: DatasetConfig,
    split: str,
    preprocessor: Optional[Callable] = None,
    postprocessors: Optional[List[Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
) -> AudioDataset:
    """
    Dataset entry point that supports both local and GS paths, with transformations.

    1. Loads datasets
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.

    Parameters
    ----------
    data_config : DataConfig
        Configuration for the dataset
    preprocessor : Optional[Callable]
        Optional preprocessor function
    split : str
        Which split of the dataset to load.

    Returns
    -------
    AudioDataset
        An instance of the dataset with the specified transformations applied.
    """

    ds = AnimalSpeak(data_config)
    ds._load(split)

    # Check if the dataset CSV path is a gs:// path
    # df = _get_dataset_from_name(data_config.dataset_name, split)

    # metadata = {}

    # if data_config.transformations:
    #     for cfg in data_config.transformations:
    #         transform = transform_from_config(cfg)
    #         df, md = transform(df)

            # TODO (milad): hacky but let's think about it
            # TODO (test if keys already exist and shout?)
    #        if md:
    #            metadata.update(md)

    # TODO (milad) transform API should be AudioDataset -> AudioDataset not df->df

    return ds
    
    # AudioDataset(
    #     df=df,
    #     data_config=data_config,
    #     preprocessor=preprocessor,
    #     metadata=metadata,
    #     postprocessors=postprocessors,
    # )


#######################################################################################
# ANYTHING BELOW IS A WIP FOR DATASET ABSTRACTION
#######################################################################################


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
    load(split: Literal["train", "validation"]) -> pd.DataFrame
        Required method to load a specific split of the dataset.
    __len__() -> int
        Required method to return the number of samples in the dataset.
    __iter__() -> Iterator[Dict[str, Any]]
        Required method to iterate over the samples in the dataset.
    __getitem__(idx: int) -> Dict[str, Any]
        Required method to get a specific sample from the dataset.
    """

    def __init__(self, dataset_config: DatasetConfig = None) -> None:
        """A DatasetConfig can be passed to the constructor to, for instance,
        apply transformations to the dataset during instanciation.

        Parameters
        ----------
        dataset_config : DatasetConfig
            The configuration for the dataset.
        """
        self.dataset_config = dataset_config


    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Dataset metadata and configuration.

        Returns
        -------
        DatasetInfo
            Object containing dataset metadata like name, version, paths, etc.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Sequence[Any]:
        """The dataset as a sequence of objects.

        Returns
        -------
        Sequence[Any]
            The dataset as a sequence of objects.
        """
        raise NotImplementedError

    @abstractmethod
    def _load(self, split: str) -> Sequence[Any]:
        """Load one split of the dataset. It should apply transformations if any in self.dataset_config.

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

    @abstractmethod
    def _apply_transformations(self, transformations: list[RegisteredTransformConfigs]) -> None:
        """Apply the given list of transformations to the dataset. This should be an in place operation.
        Ideally, this should be called either by the constructor or by the load method.

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
