"""Tests for the BirdNET model (TensorFlow wrapper)."""

import pytest
import torch

from avex.models.birdnet import Model as BirdNetModel


@pytest.fixture(scope="session")
def birdnet_model() -> BirdNetModel:
    """Create a BirdNET model for testing (session-scoped, shared across tests).

    Returns:
        BirdNetModel: A configured BirdNET model for testing.
    """
    return BirdNetModel(num_classes=10, device="cpu")


@pytest.fixture(scope="session")
def birdnet_model_no_classifier() -> BirdNetModel:
    """Create a BirdNET model without classifier for testing (session-scoped).

    Returns:
        BirdNetModel: A configured BirdNET model without classifier.
    """
    return BirdNetModel(num_classes=None, device="cpu")


@pytest.fixture
def audio_input() -> torch.Tensor:
    """Create realistic audio input tensor - 5 seconds at 48kHz.

    Returns:
        torch.Tensor: Audio input tensor with shape (2, 240000).
    """
    return torch.randn(2, 48000 * 5)


def test_birdnet_model_initialization(birdnet_model: BirdNetModel) -> None:
    """Test BirdNET model initialization."""
    assert birdnet_model.num_classes == 10
    assert birdnet_model.num_species > 0
    assert birdnet_model.classifier is not None
    assert birdnet_model.classifier.in_features == birdnet_model.num_species
    assert birdnet_model.classifier.out_features == 10


def test_birdnet_model_no_classifier(birdnet_model_no_classifier: BirdNetModel) -> None:
    """Test BirdNET model without classifier."""
    assert birdnet_model_no_classifier.num_classes is None
    assert birdnet_model_no_classifier.classifier is None


def test_extract_embeddings_aggregations(birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
    """Test extract_embeddings with different aggregation methods."""
    # Test mean aggregation
    result_mean = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")
    assert isinstance(result_mean, torch.Tensor)
    assert result_mean.shape == (2, 1024)
    assert not torch.isnan(result_mean).any()
    assert not torch.isinf(result_mean).any()

    # Test max aggregation
    result_max = birdnet_model.extract_embeddings(x=audio_input, aggregation="max")
    assert isinstance(result_max, torch.Tensor)
    assert result_max.shape == (2, 1024)
    assert not torch.isnan(result_max).any()

    # Test none aggregation (for sequence probes)
    result_none = birdnet_model.extract_embeddings(x=audio_input, aggregation="none")
    assert isinstance(result_none, list)
    assert len(result_none) == 2
    for item in result_none:
        assert item.dim() == 3
        assert item.shape[2] == 1024
        assert not torch.isnan(item).any()

    # Test cls_token aggregation
    result_cls = birdnet_model.extract_embeddings(x=audio_input, aggregation="cls_token")
    assert isinstance(result_cls, torch.Tensor)
    assert result_cls.shape == (2, 1024)


def test_extract_embeddings_dict_input(birdnet_model: BirdNetModel) -> None:
    """Test extract_embeddings with dictionary input."""
    dict_input = {"raw_wav": torch.randn(2, 48000 * 5)}
    result = birdnet_model.extract_embeddings(x=dict_input, aggregation="mean")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 1024)
    assert not torch.isnan(result).any()


def test_extract_embeddings_invalid_aggregation(birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
    """Test extract_embeddings with invalid aggregation method."""
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        birdnet_model.extract_embeddings(x=audio_input, aggregation="invalid_method")


def test_forward_method(
    birdnet_model: BirdNetModel,
    birdnet_model_no_classifier: BirdNetModel,
    audio_input: torch.Tensor,
) -> None:
    """Test the forward method with and without classifier."""
    # With classifier
    result = birdnet_model.forward(audio_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 10)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # Without classifier
    result_no_clf = birdnet_model_no_classifier.forward(audio_input)
    assert isinstance(result_no_clf, torch.Tensor)
    assert result_no_clf.shape == (2, birdnet_model_no_classifier.num_species)
    assert not torch.isnan(result_no_clf).any()


def test_species_mapping(birdnet_model: BirdNetModel) -> None:
    """Test species index mapping methods."""
    species_name = birdnet_model.idx_to_species(0)
    assert isinstance(species_name, str)
    assert len(species_name) > 0

    idx = birdnet_model.species_to_idx(species_name)
    assert idx == 0

    with pytest.raises(ValueError):
        birdnet_model.species_to_idx("invalid_species_that_does_not_exist_12345")


def test_return_features_only() -> None:
    """Test return_features_only parameter handling."""
    # return_features_only=True should set num_classes to None
    model = BirdNetModel(return_features_only=True, device="cpu")
    assert model.return_features_only is True
    assert model.num_classes is None
    assert model.classifier is None

    # return_features_only=False with explicit num_classes
    model2 = BirdNetModel(return_features_only=False, num_classes=20, device="cpu")
    assert model2.return_features_only is False
    assert model2.num_classes == 20
    assert model2.classifier is not None

    # return_features_only=True overrides num_classes
    model3 = BirdNetModel(return_features_only=True, num_classes=50, device="cpu")
    assert model3.num_classes is None
