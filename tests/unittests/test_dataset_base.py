import pytest

from esp_data_temp.config import DatasetConfig
from esp_data_temp.datasets import (
    Dataset,
    DatasetInfo,
    dataset_from_config,
    list_registered_datasets,
    print_registered_datasets,
)


@pytest.fixture
def temp_pandas_dataset() -> Dataset:
    """Fixture providing a temporary pandas dataset.

    Returns
    -------
    Dataset
        A temporary pandas dataset.
    """
    import tempfile

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(b"column1,column2\nvalue1,value2\n")
        tmp_file_path = tmp_file.name

    return tmp_file_path


def test_print_registered_datasets() -> None:
    """Test if the print_registered_datasets function prints the correct information."""
    # Capture the output of the print function
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output

    print_registered_datasets()

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Check if the output contains expected dataset names
    output = captured_output.getvalue()
    assert "animalspeak" in output
    assert "barkley_canyon" in output


def test_list_registered_datasets() -> None:
    """Test if the list_registered_datasets function
    returns the correct dataset names.
    """
    dataset_names = list_registered_datasets()
    assert len(dataset_names) > 0  # Ensure there is at least one dataset registered
    assert "animalspeak" in dataset_names
    assert "barkley_canyon" in dataset_names


def test_load_dataset() -> None:
    """Test if the load_dataset function loads the dataset correctly."""
    cfg = DatasetConfig(
        dataset_name="animalspeak", split="validation", audio_path_col="gs_path"
    )
    dataset = dataset_from_config(cfg)
    assert dataset is not None
    assert isinstance(
        dataset, Dataset
    )  # Check if the loaded dataset is of correct type
    assert hasattr(dataset, "info")  # Check if dataset has info property
    assert dataset.info.name == "animalspeak"
    assert isinstance(dataset.info.version, str)
    assert len(dataset.info.split_paths) > 0  # Ensure there are split paths available


def test_dataset_info(temp_pandas_dataset: Dataset) -> None:
    """Test if the DatasetInfo class initializes correctly."""
    # make temporary datasets at /tmp_path
    dataset_info = DatasetInfo(
        name="animalspeak",
        owner="david",
        split_paths={"train": temp_pandas_dataset},
        version="0.1.0",
        description="Test dataset",
        sources=["source1", "source2"],
        license="MIT",
    )

    assert dataset_info.name == "animalspeak"
    assert dataset_info.owner == "david"
    assert dataset_info.split_paths["train"] == temp_pandas_dataset
    assert dataset_info.version == "0.1.0"
    assert dataset_info.description == "Test dataset"
    assert dataset_info.sources == ["source1", "source2"]
    assert dataset_info.license == "MIT"

    with pytest.raises(ValueError):
        # Test if ValueError is raised when split_paths is empty
        DatasetInfo(
            name="animalspeak",
            owner="david",
            split_paths={},
            version="0.1.0",
            description="Test dataset",
            sources=["source1", "source2"],
            license="MIT",
        )
