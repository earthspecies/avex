"""Tests for the SurfPerch model (TensorFlow wrapper)."""

from typing import Dict

import pytest
import torch

from representation_learning.models.surfperch import PerchModel


@pytest.fixture(scope="session")
def surfperch_model() -> PerchModel:
    """Create a SurfPerch model for testing (session-scoped, shared across tests).

    Returns:
        PerchModel: A configured SurfPerch model for testing.
    """
    return PerchModel(num_classes=10, device="cpu")


@pytest.fixture(scope="session")
def surfperch_model_no_classifier() -> PerchModel:
    """Create a SurfPerch model without classifier for testing (session-scoped).

    Returns:
        PerchModel: A configured SurfPerch model without classifier.
    """
    return PerchModel(num_classes=None, device="cpu")


@pytest.fixture
def audio_input() -> torch.Tensor:
    """Create realistic audio input tensor - 5 seconds at 32kHz.

    Returns:
        torch.Tensor: Audio input tensor with shape (2, 160000).
    """
    return torch.randn(2, 32000 * 5)


def test_surfperch_model_initialization(surfperch_model: PerchModel) -> None:
    """Test SurfPerch model initialization."""
    assert surfperch_model.num_classes == 10
    assert surfperch_model.embedding_dim == 1280
    assert surfperch_model.classifier is not None
    assert surfperch_model.classifier.in_features == 1280
    assert surfperch_model.classifier.out_features == 10
    assert surfperch_model.device == "cpu"
    assert surfperch_model.target_sr == 32000
    assert surfperch_model.window_samples == 160000


def test_surfperch_model_no_classifier(surfperch_model_no_classifier: PerchModel) -> None:
    """Test SurfPerch model without classifier."""
    assert surfperch_model_no_classifier.num_classes is None
    assert surfperch_model_no_classifier.classifier is None


def test_extract_embeddings_aggregations(surfperch_model: PerchModel, audio_input: torch.Tensor) -> None:
    """Test extract_embeddings with different aggregation methods."""
    # Test mean aggregation
    result_mean = surfperch_model.extract_embeddings(x=audio_input, aggregation="mean")
    assert isinstance(result_mean, torch.Tensor)
    assert result_mean.shape == (2, 1280)
    assert not torch.isnan(result_mean).any()
    assert not torch.isinf(result_mean).any()

    # Test max aggregation
    result_max = surfperch_model.extract_embeddings(x=audio_input, aggregation="max")
    assert isinstance(result_max, torch.Tensor)
    assert result_max.shape == (2, 1280)
    assert not torch.isnan(result_max).any()

    # Test none aggregation (for sequence probes)
    result_none = surfperch_model.extract_embeddings(x=audio_input, aggregation="none")
    assert isinstance(result_none, list)
    assert len(result_none) == 1
    for item in result_none:
        assert item.dim() == 3
        assert item.shape[2] == 1280
        assert not torch.isnan(item).any()

    # Test cls_token aggregation
    result_cls = surfperch_model.extract_embeddings(x=audio_input, aggregation="cls_token")
    assert isinstance(result_cls, torch.Tensor)
    assert result_cls.shape == (2, 1280)


def test_extract_embeddings_dict_input(surfperch_model: PerchModel) -> None:
    """Test extract_embeddings with dictionary input."""
    dict_input: Dict[str, torch.Tensor] = {"raw_wav": torch.randn(2, 32000 * 5)}
    result = surfperch_model.extract_embeddings(x=dict_input, aggregation="mean")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 1280)
    assert not torch.isnan(result).any()


def test_extract_embeddings_invalid_aggregation(surfperch_model: PerchModel, audio_input: torch.Tensor) -> None:
    """Test extract_embeddings with invalid aggregation method."""
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        surfperch_model.extract_embeddings(x=audio_input, aggregation="invalid_method")


def test_forward_method(
    surfperch_model: PerchModel,
    surfperch_model_no_classifier: PerchModel,
    audio_input: torch.Tensor,
) -> None:
    """Test the forward method with and without classifier."""
    # With classifier
    with torch.no_grad():
        result = surfperch_model.forward(audio_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 10)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # Without classifier
    with torch.no_grad():
        result_no_clf = surfperch_model_no_classifier.forward(audio_input)
    assert isinstance(result_no_clf, torch.Tensor)
    assert result_no_clf.shape == (2, 1280)
    assert not torch.isnan(result_no_clf).any()


def test_return_features_only() -> None:
    """Test return_features_only parameter handling."""
    # return_features_only=True should set num_classes to None
    model = PerchModel(return_features_only=True, device="cpu")
    assert model.return_features_only is True
    assert model.num_classes is None
    assert model.classifier is None

    # return_features_only=False with explicit num_classes
    model2 = PerchModel(return_features_only=False, num_classes=20, device="cpu")
    assert model2.return_features_only is False
    assert model2.num_classes == 20
    assert model2.classifier is not None

    # return_features_only=True overrides num_classes
    model3 = PerchModel(return_features_only=True, num_classes=50, device="cpu")
    assert model3.num_classes is None
