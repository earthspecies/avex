"""
Tests for the Perch model (TensorFlow wrapper).

Tests the extract_embeddings interface with aggregation methods
and sequence probe compatibility using real model instances.
"""

from typing import Dict

import pytest
import torch

from representation_learning.models.perch import PerchModel
from tests.utils.test_utils import create_cleanup_hooks_fixture


class TestPerchModel:
    """Test suite for Perch model."""

    # Cleanup hooks after each test for model fixtures
    cleanup_hooks = create_cleanup_hooks_fixture(model_fixture_name="perch_model")

    @pytest.fixture(scope="session")
    def perch_model(self) -> PerchModel:
        """Create a Perch model for testing (session-scoped, shared across tests).

        Returns:
            PerchModel: A configured Perch model for testing.
        """
        return PerchModel(num_classes=10, device="cpu")

    @pytest.fixture(scope="session")
    def perch_model_no_classifier(self) -> PerchModel:
        """Create a Perch model without classifier for testing (session-scoped).

        Returns:
            PerchModel: A configured Perch model without classifier.
        """
        return PerchModel(num_classes=0, device="cpu")

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Create realistic audio input tensor.

        Returns:
            torch.Tensor: Audio input tensor with shape (2, 160000) - 5 seconds at 32kHz.
        """
        # Generate 5 seconds of audio at 32kHz (Perch's sample rate)
        return torch.randn(2, 32000 * 5)

    @pytest.fixture
    def dict_input(self) -> Dict[str, torch.Tensor]:
        """Create dictionary input with raw_wav.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with raw_wav key.
        """
        return {"raw_wav": torch.randn(2, 32000 * 5)}

    @pytest.fixture
    def padding_mask(self) -> torch.Tensor:
        """Create padding mask.

        Returns:
            torch.Tensor: Padding mask tensor with shape (2, 160000).
        """
        return torch.ones(2, 32000 * 5, dtype=torch.bool)

    def test_perch_model_initialization(self, perch_model: PerchModel) -> None:
        """Test Perch model initialization."""
        assert perch_model.num_classes == 10
        assert perch_model.embedding_dim == 1280
        assert perch_model.classifier is not None
        assert perch_model.classifier.in_features == 1280
        assert perch_model.classifier.out_features == 10
        assert perch_model.device == "cpu"
        assert perch_model.target_sr == 32000
        assert perch_model.window_samples == 160000

    def test_perch_model_no_classifier(self, perch_model_no_classifier: PerchModel) -> None:
        """Test Perch model without classifier."""
        assert perch_model_no_classifier.num_classes == 0
        assert perch_model_no_classifier.classifier is None

    def test_extract_embeddings_aggregation_mean(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='mean' (default)."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)  # (batch_size, embedding_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_aggregation_none(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='none' for sequence probes."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="none")

        assert isinstance(result, list)
        assert len(result) == 1  # Single layer model

        # Each item should be 3D tensor (B, 1, 1280)
        for item in result:
            assert isinstance(item, torch.Tensor)
            assert item.dim() == 3  # (B, 1, 1280)
            assert item.shape[0] == 2  # Batch dimension
            assert item.shape[2] == 1280  # Embedding dimension
            assert not torch.isnan(item).any()
            assert not torch.isinf(item).any()

    def test_extract_embeddings_aggregation_max(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='max'."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="max")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_aggregation_cls_token(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='cls_token'."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="cls_token")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_dict_input(self, perch_model: PerchModel, dict_input: Dict[str, torch.Tensor]) -> None:
        """Test extract_embeddings with dictionary input."""
        result = perch_model.extract_embeddings(x=dict_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_invalid_aggregation(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with invalid aggregation method."""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            perch_model.extract_embeddings(x=audio_input, aggregation="invalid_method")

    def test_extract_embeddings_sequence_probe_compatibility(
        self, perch_model: PerchModel, audio_input: torch.Tensor
    ) -> None:
        """Test that embeddings are compatible with sequence probes."""
        # Test with aggregation="none" for sequence probes
        result = perch_model.extract_embeddings(x=audio_input, aggregation="none")

        # Should return list of 3D tensors
        assert isinstance(result, list)
        assert len(result) == 1  # Single layer model

        for embedding_tensor in result:
            assert embedding_tensor.dim() == 3  # (B, 1, 1280)
            assert embedding_tensor.shape[0] == 2  # Batch dimension
            assert embedding_tensor.shape[2] == 1280  # Embedding dimension
            assert not torch.isnan(embedding_tensor).any()
            assert not torch.isinf(embedding_tensor).any()

    def test_extract_embeddings_device_consistency(self, audio_input: torch.Tensor) -> None:
        """Test that embeddings are returned on the right device."""
        # Test on CPU
        model_cpu = PerchModel(num_classes=10, device="cpu")
        result_cpu = model_cpu.extract_embeddings(x=audio_input, aggregation="mean")
        assert result_cpu.device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = PerchModel(num_classes=10, device="cuda")
            result_cuda = model_cuda.extract_embeddings(x=audio_input, aggregation="mean")
            assert result_cuda.device.type == "cuda"

    def test_extract_embeddings_padding_mask_handling(
        self,
        perch_model: PerchModel,
        audio_input: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        """Test that padding_mask is handled correctly (even though unused)."""
        # Test with padding_mask (should not affect output)
        result_with_mask = perch_model.extract_embeddings(x=audio_input, aggregation="mean", padding_mask=padding_mask)

        result_without_mask = perch_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Results should be identical since padding_mask is unused
        assert torch.allclose(result_with_mask, result_without_mask, atol=1e-6)

    def test_extract_embeddings_consistency(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings produces consistent results."""
        # Test multiple calls with same input
        result1 = perch_model.extract_embeddings(x=audio_input, aggregation="mean")
        result2 = perch_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Results should be identical (deterministic)
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_forward_method(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test the forward method with classifier."""
        with torch.no_grad():
            result = perch_model.forward(audio_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10)  # (batch_size, num_classes)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_method_no_classifier(
        self, perch_model_no_classifier: PerchModel, audio_input: torch.Tensor
    ) -> None:
        """Test the forward method without classifier."""
        with torch.no_grad():
            result = perch_model_no_classifier.forward(audio_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)  # (batch_size, embedding_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_device_movement(self, perch_model: PerchModel) -> None:
        """Test device movement methods."""
        # Test moving to CUDA
        if torch.cuda.is_available():
            model_cuda = perch_model.cuda()
            assert model_cuda.device == "cpu"  # String attribute doesn't change
            if model_cuda.classifier is not None:
                assert next(model_cuda.classifier.parameters()).device.type == "cuda"

        # Test moving to CPU
        model_cpu = perch_model.cpu()
        assert model_cpu.device == "cpu"
        if model_cpu.classifier is not None:
            assert next(model_cpu.classifier.parameters()).device.type == "cpu"

    def test_model_attributes(self, perch_model: PerchModel) -> None:
        """Test that model attributes are correctly set."""
        # Test model attributes
        assert hasattr(perch_model, "embedding_dim")
        assert hasattr(perch_model, "num_classes")
        assert hasattr(perch_model, "classifier")
        assert hasattr(perch_model, "device")
        assert hasattr(perch_model, "target_sr")
        assert hasattr(perch_model, "window_samples")
        assert perch_model.embedding_dim == 1280
        assert perch_model.num_classes == 10

    def test_model_methods(self, perch_model: PerchModel) -> None:
        """Test that model methods exist and are callable."""
        # Test that methods exist
        assert hasattr(perch_model, "extract_embeddings")
        assert hasattr(perch_model, "forward")
        assert hasattr(perch_model, "extract_features")
        assert callable(perch_model.extract_embeddings)
        assert callable(perch_model.forward)
        assert callable(perch_model.extract_features)

    def test_model_initialization_edge_cases(self) -> None:
        """Test model initialization with edge cases."""
        # Test with very large num_classes
        model_large = PerchModel(num_classes=1000, device="cpu")
        assert model_large.num_classes == 1000
        assert model_large.classifier is not None
        assert model_large.classifier.out_features == 1000

    def test_model_embedding_consistency_across_batches(self, perch_model: PerchModel) -> None:
        """Test that embeddings are consistent across different batch sizes."""
        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 32000 * 5)
            result = perch_model.extract_embeddings(x=input_tensor, aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, 1280)
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

    def test_embedding_dimension(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test that embeddings are correctly 1280-dimensional."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="mean")

        assert result.shape == (2, 1280)
        # Verify it's not the wrong dimension
        assert result.shape[1] == 1280

    def test_embedding_values_range(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test that embedding values are in reasonable range."""
        result = perch_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Embeddings should be finite and not all zeros
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_forward_output_range(self, perch_model: PerchModel, audio_input: torch.Tensor) -> None:
        """Test that forward output values are reasonable."""
        with torch.no_grad():
            result = perch_model.forward(audio_input)

        # Output should be finite
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        # Output should have reasonable values (not all zeros or extreme)
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_extract_embeddings_single_sample(self, perch_model: PerchModel) -> None:
        """Test extract_embeddings with single sample."""
        single_audio = torch.randn(1, 32000 * 5)
        result = perch_model.extract_embeddings(x=single_audio, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1280)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_return_features_only_true(self) -> None:
        """Test Perch model with return_features_only=True."""
        model = PerchModel(return_features_only=True, device="cpu")

        assert model.return_features_only is True
        assert model.num_classes == 0
        assert model.classifier is None

    def test_return_features_only_false_with_num_classes(self) -> None:
        """Test Perch model with return_features_only=False and explicit num_classes."""
        model = PerchModel(return_features_only=False, num_classes=20, device="cpu")

        assert model.return_features_only is False
        assert model.num_classes == 20
        assert model.classifier is not None
        assert model.classifier.out_features == 20

    def test_return_features_only_false_without_num_classes(self) -> None:
        """Test Perch model with return_features_only=False and num_classes=None (defaults to 0)."""
        model = PerchModel(return_features_only=False, device="cpu")

        assert model.return_features_only is False
        assert model.num_classes == 0  # None defaults to 0
        assert model.classifier is None

    def test_return_features_only_overrides_num_classes(self) -> None:
        """Test that return_features_only=True overrides num_classes parameter."""
        # Even if num_classes is provided, return_features_only=True should set it to 0
        model = PerchModel(return_features_only=True, num_classes=50, device="cpu")

        assert model.return_features_only is True
        assert model.num_classes == 0  # Should be overridden
        assert model.classifier is None

    def test_return_features_only_forward_pass(self, audio_input: torch.Tensor) -> None:
        """Test forward pass with return_features_only=True."""
        model = PerchModel(return_features_only=True, device="cpu")

        with torch.no_grad():
            result = model.forward(audio_input)

        # Should return embeddings (no classifier)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)  # (batch_size, embedding_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_return_features_only_extract_embeddings(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with return_features_only=True."""
        model = PerchModel(return_features_only=True, device="cpu")

        result = model.extract_embeddings(x=audio_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1280)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


if __name__ == "__main__":
    pytest.main([__file__])
