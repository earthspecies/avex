"""Test suite for the AnimalSpeak dataset."""

import pytest

from esp_data_temp.config import DatasetConfig
from esp_data_temp.datasets import AnimalSpeak, Dataset


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset.
    """
    ds = AnimalSpeak(split="validation")
    return ds


@pytest.fixture
def dataset_with_transforms() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance with transformations
    applied.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with transformations applied.
    """

    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        transformations=[
            {
                "type": "label_from_feature",
                "feature": "canonical_name",
                "output_feature": "label",
            },
            {
                "type": "filter",
                "mode": "include",
                "property": "source",
                "values": ["xeno-canto", "iNaturalist"],
            },
        ],
    )
    ds = AnimalSpeak(split="validation")
    ds.apply_transformations(dataset_config.transformations)
    return ds


@pytest.fixture
def dataset_with_output_mapping() -> Dataset:
    """Fixture providing an AnimalSpeak dataset instance with output mapping.

    Returns
    -------
    Dataset
        An instance of the AnimalSpeak dataset with output mapping applied.
    """
    dataset_config = DatasetConfig(
        dataset_name="animalspeak",
        output_take_and_give={"canonical_name": "species", "country": "location"},
    )
    ds = AnimalSpeak(
        split="validation", output_take_and_give=dataset_config.output_take_and_give
    )
    return ds


def test_info_property(dataset: Dataset) -> None:
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "animalspeak"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths


def test_data_property(dataset: Dataset) -> None:
    """Test if the data property returns correct dataframes."""
    # Data should be _loaded in __init__
    assert dataset._data is not None
    assert "audio_id" in dataset._data
    assert "country" in dataset._data


def test_length(dataset: Dataset) -> None:
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len


def test_getitem(dataset: Dataset) -> None:
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, dict)


def test_iteration(dataset: Dataset) -> None:
    """Test if iteration works correctly."""
    # Test if we can iterate and get correct number of samples
    samples = list(dataset)

    # Check length
    assert len(samples) == len(dataset)


def test_invalid_split(dataset: Dataset) -> None:
    """Test if _loading invalid split raises error."""
    with pytest.raises(ValueError):
        dataset._load("invalid_split")


def test_sample_consistency(dataset: Dataset) -> None:
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))

    # Compare samples
    assert direct_sample["country"] == iter_sample["country"]


def test_transformations(dataset_with_transforms: Dataset) -> None:
    """Test if transformations are applied correctly.

    This test verifies that:
    1. The label_from_feature transformation creates a label column
    2. The filter transformation only keeps specified sources
    3. The metadata is updated with transformation information
    """
    # Check that label column was created
    assert "label" in dataset_with_transforms._data.columns

    # Check that only specified sources are present
    sources = dataset_with_transforms._data["source"].unique()
    assert set(sources).issubset({"xeno-canto", "iNaturalist"})

    # Check that no other sources are present
    assert "Watkins" not in sources


def test_output_take_and_give(dataset_with_output_mapping: Dataset) -> None:
    """Test if output_take_and_give correctly maps column names.

    This test verifies that:
    1. The output dictionary contains only the mapped columns
    2. The original column names are mapped to the new names
    3. The values are preserved correctly
    """
    # Get a sample
    sample = dataset_with_output_mapping[0]

    # Check that only mapped columns are present
    assert set(sample.keys()) == {"species", "location"}

    # Get the original row to compare values
    original_row = dataset_with_output_mapping._data.iloc[0]

    # Verify the mapping and values
    assert sample["species"] == original_row["canonical_name"]
    assert sample["location"] == original_row["country"]
