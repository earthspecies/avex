"""Registry for ESP datasets."""

import pathlib
from typing import Optional, Protocol, TypeVar, Type

import semver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data_temp.dataset import GSPath, DatasetInfo



class RegisteredDatasetProtocol(Protocol):
    """Protocol for registered dataset classes."""
    info: DatasetInfo


# Type variable for registered dataset classes
RegisteredDataset = TypeVar("RegisteredDataset", bound=RegisteredDatasetProtocol)


class DatasetRegistry:
    """A registry for all registered datasets. This is a singleton class."""

    def __init__(self):
        """Initialize an empty registry."""
        self.datasets: dict[str, Type[RegisteredDataset]] = {}

    def register(self, dataset_class: Type[RegisteredDataset]) -> None:
        """Register a dataset class.
        
        Parameters
        ----------
        dataset_class : Type[RegisteredDataset]
            The dataset class to register
        """
        # Create a temporary instance to get the info
        temp_instance = dataset_class()
        name = temp_instance.info.name
        self.datasets[name] = dataset_class

    def list(self) -> list[str]:
        """List all registered datasets.

        Returns
        -------
        list[str]
            List of dataset names
        """
        return list(self.datasets.keys())

    def print(self) -> None:
        """Print all registered datasets."""
        for dataset_class in self.datasets.values():
            # Create temporary instance to access info
            temp_instance = dataset_class()
            print(temp_instance.info.model_dump_json(indent=2))


# Global registry instance
registry = DatasetRegistry()


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
    
    registry.register(cls)
    return cls
