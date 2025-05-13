"""Test suite for the AnimalSpeak dataset."""

import pytest
import numpy as np
from esp_data_temp.registered_datasets import AnimalSpeak


@pytest.fixture
def dataset():
    """Fixture providing an AnimalSpeak dataset instance."""
    ds = AnimalSpeak()
    ds.load("validation")
    return ds


def test_info_property(dataset):
    """Test if the info property returns correct metadata."""
    assert dataset.info.name == "animalspeak"
    assert dataset.info.version == "0.1.0"
    assert "train" in dataset.info.split_paths
    assert "validation" in dataset.info.split_paths


def test_data_property(dataset):
    """Test if the data property returns correct dataframes."""
    # Data should be loaded in __init__
    assert dataset._data is not None
    assert "audio_id" in dataset._data
    assert "country" in dataset._data


def test_length(dataset):
    """Test if __len__ returns correct counts."""
    # Length should be sum of all splits
    expected_len = dataset._data.shape[0]
    assert len(dataset) == expected_len


def test_getitem(dataset):
    """Test if __getitem__ returns correct sample format."""
    # Get first sample
    sample = dataset[0]
    


def test_iteration(dataset):
    """Test if iteration works correctly."""
    # Test if we can iterate and get correct number of samples
    samples = list(dataset)
    
    # Check length
    assert len(samples) == len(dataset)
    


def test_invalid_split(dataset):
    """Test if loading invalid split raises error."""
    with pytest.raises(ValueError):
        dataset.load("invalid_split")



def test_sample_consistency(dataset):
    """Test if samples are consistent when accessed multiple ways."""
    # Get same sample through different methods
    direct_sample = dataset[0]
    iter_sample = next(iter(dataset))
    
    # Compare samples
    assert direct_sample["country"] == iter_sample["country"]