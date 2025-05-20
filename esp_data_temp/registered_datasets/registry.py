from typing import Protocol, TypeVar


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
