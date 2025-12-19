"""
Tests for the BirdNET model (TensorFlow wrapper).

Tests the extract_embeddings interface with aggregation methods
and sequence probe compatibility using real model instances.
"""

from typing import Dict

import pytest
import torch

from representation_learning.models.birdnet import Model as BirdNetModel
from tests.utils.test_utils import create_cleanup_hooks_fixture


class TestBirdNetModel:
    """Test suite for BirdNET model."""

    # Cleanup hooks after each test for model fixtures
    cleanup_hooks = create_cleanup_hooks_fixture(model_fixture_name="birdnet_model")

    @pytest.fixture(scope="session")
    def _base_model(self) -> BirdNetModel:
        """Create a base BirdNET model to get num_species (session-scoped).

        Returns:
            BirdNetModel: A base BirdNET model for getting num_species.
        """
        return BirdNetModel(num_classes=0, device="cpu")

    @pytest.fixture(scope="session")
    def birdnet_model(self, _base_model: BirdNetModel) -> BirdNetModel:
        """Create a BirdNET model for testing (session-scoped, shared across tests).

        Returns:
            BirdNetModel: A configured BirdNET model for testing.
        """
        return BirdNetModel(num_classes=10, device="cpu")

    @pytest.fixture(scope="session")
    def birdnet_model_no_classifier(self) -> BirdNetModel:
        """Create a BirdNET model without classifier for testing (session-scoped).

        Returns:
            BirdNetModel: A configured BirdNET model without classifier.
        """
        return BirdNetModel(num_classes=0, device="cpu")

    @pytest.fixture(scope="session")
    def birdnet_model_matching_classes(self, _base_model: BirdNetModel) -> BirdNetModel:
        """Create a BirdNET model with num_classes matching num_species (session-scoped).

        Returns:
            BirdNetModel: A configured BirdNET model with matching classes.
        """
        num_species = _base_model.num_species
        return BirdNetModel(num_classes=num_species, device="cpu")

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Create realistic audio input tensor.

        Returns:
            torch.Tensor: Audio input tensor with shape (2, 240000) - 5 seconds at 48kHz.
        """
        # Generate 5 seconds of audio at 48kHz (BirdNET's sample rate)
        return torch.randn(2, 48000 * 5)

    @pytest.fixture
    def dict_input(self) -> Dict[str, torch.Tensor]:
        """Create dictionary input with raw_wav.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with raw_wav key.
        """
        return {"raw_wav": torch.randn(2, 48000 * 5)}

    @pytest.fixture
    def padding_mask(self) -> torch.Tensor:
        """Create padding mask.

        Returns:
            torch.Tensor: Padding mask tensor with shape (2, 240000).
        """
        return torch.ones(2, 48000 * 5, dtype=torch.bool)

    def test_birdnet_model_initialization(self, birdnet_model: BirdNetModel) -> None:
        """Test BirdNET model initialization."""
        assert birdnet_model.num_classes == 10
        assert birdnet_model.num_species > 0  # Real BirdNET has many species
        assert birdnet_model.classifier is not None
        assert birdnet_model.classifier.in_features == birdnet_model.num_species
        assert birdnet_model.classifier.out_features == 10
        assert birdnet_model.device == "cpu"

    def test_birdnet_model_no_classifier(self, birdnet_model_no_classifier: BirdNetModel) -> None:
        """Test BirdNET model without classifier."""
        assert birdnet_model_no_classifier.num_classes == 0
        assert birdnet_model_no_classifier.classifier is None

    def test_birdnet_model_matching_classes(self, birdnet_model_matching_classes: BirdNetModel) -> None:
        """Test BirdNET model when num_classes matches num_species."""
        assert birdnet_model_matching_classes.num_classes == birdnet_model_matching_classes.num_species
        assert birdnet_model_matching_classes.classifier is None  # No need for additional classifier

    def test_extract_embeddings_aggregation_mean(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='mean' (default)."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1024)  # (batch_size, embedding_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_aggregation_none(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='none' for sequence probes."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="none")

        assert isinstance(result, list)
        assert len(result) == 2  # One per batch item

        # Each item should be 3D tensor (1, N, 1024)
        for item in result:
            assert isinstance(item, torch.Tensor)
            assert item.dim() == 3  # (1, N, 1024)
            assert item.shape[0] == 1  # Batch dimension
            assert item.shape[2] == 1024  # Embedding dimension
            assert not torch.isnan(item).any()
            assert not torch.isinf(item).any()

    def test_extract_embeddings_aggregation_max(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='max'."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="max")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1024)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_aggregation_cls_token(
        self, birdnet_model: BirdNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='cls_token'."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="cls_token")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1024)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_dict_input(
        self, birdnet_model: BirdNetModel, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test extract_embeddings with dictionary input."""
        result = birdnet_model.extract_embeddings(x=dict_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1024)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_extract_embeddings_invalid_aggregation(
        self, birdnet_model: BirdNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings with invalid aggregation method."""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            birdnet_model.extract_embeddings(x=audio_input, aggregation="invalid_method")

    def test_extract_embeddings_sequence_probe_compatibility(
        self, birdnet_model: BirdNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that embeddings are compatible with sequence probes."""
        # Test with aggregation="none" for sequence probes
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="none")

        # Should return list of 3D tensors
        assert isinstance(result, list)
        assert len(result) == 2  # One per batch item

        for embedding_tensor in result:
            assert embedding_tensor.dim() == 3  # (1, N, 1024)
            assert embedding_tensor.shape[0] == 1  # Batch dimension
            assert embedding_tensor.shape[2] == 1024  # Embedding dimension
            assert not torch.isnan(embedding_tensor).any()
            assert not torch.isinf(embedding_tensor).any()

    def test_extract_embeddings_device_consistency(self, audio_input: torch.Tensor) -> None:
        """Test that embeddings are returned on the right device."""
        # Test on CPU
        model_cpu = BirdNetModel(num_classes=10, device="cpu")
        result_cpu = model_cpu.extract_embeddings(x=audio_input, aggregation="mean")
        assert result_cpu.device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = BirdNetModel(num_classes=10, device="cuda")
            result_cuda = model_cuda.extract_embeddings(x=audio_input, aggregation="mean")
            assert result_cuda.device.type == "cuda"

    def test_extract_embeddings_padding_mask_handling(
        self,
        birdnet_model: BirdNetModel,
        audio_input: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        """Test that padding_mask is handled correctly (even though unused)."""
        # Test with padding_mask (should not affect output)
        result_with_mask = birdnet_model.extract_embeddings(
            x=audio_input, aggregation="mean", padding_mask=padding_mask
        )

        result_without_mask = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Results should be identical since padding_mask is unused
        assert torch.allclose(result_with_mask, result_without_mask, atol=1e-6)

    def test_extract_embeddings_consistency(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings produces consistent results."""
        # Test multiple calls with same input
        result1 = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")
        result2 = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Results should be identical (deterministic)
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_forward_method(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test the forward method with classifier."""
        result = birdnet_model.forward(audio_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10)  # (batch_size, num_classes)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_method_no_classifier(
        self, birdnet_model_no_classifier: BirdNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test the forward method without classifier."""
        result = birdnet_model_no_classifier.forward(audio_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, birdnet_model_no_classifier.num_species)  # (batch_size, num_species)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_device_movement(self, birdnet_model: BirdNetModel) -> None:
        """Test device movement methods."""
        # Test moving to CUDA
        if torch.cuda.is_available():
            model_cuda = birdnet_model.cuda()
            assert model_cuda.device.type == "cuda"
            if model_cuda.classifier is not None:
                assert next(model_cuda.classifier.parameters()).device.type == "cuda"

        # Test moving to CPU
        model_cpu = birdnet_model.cpu()
        assert model_cpu.device.type == "cpu"
        if model_cpu.classifier is not None:
            assert next(model_cpu.classifier.parameters()).device.type == "cpu"

    def test_species_mapping(self, birdnet_model: BirdNetModel) -> None:
        """Test species index mapping methods."""
        # Test idx_to_species
        species_name = birdnet_model.idx_to_species(0)
        assert isinstance(species_name, str)
        assert len(species_name) > 0

        # Test species_to_idx
        idx = birdnet_model.species_to_idx(species_name)
        assert idx == 0

        # Test invalid species
        with pytest.raises(ValueError):
            birdnet_model.species_to_idx("invalid_species_that_does_not_exist_12345")

    def test_gradient_checkpointing(self, birdnet_model: BirdNetModel) -> None:
        """Test that gradient checkpointing logs a warning."""
        # This should not raise an error, just log a warning
        birdnet_model.enable_gradient_checkpointing()

    def test_model_attributes(self, birdnet_model: BirdNetModel) -> None:
        """Test that model attributes are correctly set."""
        # Test model attributes
        assert hasattr(birdnet_model, "num_species")
        assert hasattr(birdnet_model, "num_classes")
        assert hasattr(birdnet_model, "classifier")
        assert hasattr(birdnet_model, "device")
        assert birdnet_model.num_species > 0
        assert birdnet_model.num_classes == 10

    def test_model_methods(self, birdnet_model: BirdNetModel) -> None:
        """Test that model methods exist and are callable."""
        # Test that methods exist
        assert hasattr(birdnet_model, "extract_embeddings")
        assert hasattr(birdnet_model, "forward")
        assert hasattr(birdnet_model, "_embedding_for_clip")
        assert hasattr(birdnet_model, "_infer_clip")
        assert hasattr(birdnet_model, "idx_to_species")
        assert hasattr(birdnet_model, "species_to_idx")
        assert hasattr(birdnet_model, "enable_gradient_checkpointing")
        assert callable(birdnet_model.extract_embeddings)
        assert callable(birdnet_model.forward)
        assert callable(birdnet_model._embedding_for_clip)
        assert callable(birdnet_model._infer_clip)
        assert callable(birdnet_model.idx_to_species)
        assert callable(birdnet_model.species_to_idx)
        assert callable(birdnet_model.enable_gradient_checkpointing)

    def test_model_initialization_edge_cases(self) -> None:
        """Test model initialization with edge cases."""
        # Test with very large num_classes
        model_large = BirdNetModel(num_classes=1000, device="cpu")
        assert model_large.num_classes == 1000
        assert model_large.classifier is not None
        assert model_large.classifier.out_features == 1000

    def test_model_embedding_consistency_across_batches(self, birdnet_model: BirdNetModel) -> None:
        """Test that embeddings are consistent across different batch sizes."""
        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 48000 * 5)
            result = birdnet_model.extract_embeddings(x=input_tensor, aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, 1024)
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

    def test_embedding_dimension(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test that embeddings are correctly 1024-dimensional."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")

        assert result.shape == (2, 1024)
        # Verify it's not the wrong dimension (e.g., 144000)
        assert result.shape[1] == 1024

    def test_embedding_values_range(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test that embedding values are in reasonable range."""
        result = birdnet_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Embeddings should be finite and not all zeros
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_forward_output_range(self, birdnet_model: BirdNetModel, audio_input: torch.Tensor) -> None:
        """Test that forward output values are reasonable."""
        result = birdnet_model.forward(audio_input)

        # Output should be finite
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        # Output should have reasonable values (not all zeros or extreme)
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_extract_embeddings_single_sample(self, birdnet_model: BirdNetModel) -> None:
        """Test extract_embeddings with single sample."""
        single_audio = torch.randn(1, 48000 * 5)
        result = birdnet_model.extract_embeddings(x=single_audio, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1024)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_return_features_only_true(self) -> None:
        """Test BirdNET model with return_features_only=True."""
        model = BirdNetModel(return_features_only=True, device="cpu")

        assert model.return_features_only is True
        assert model.num_classes == 0
        assert model.classifier is None

    def test_return_features_only_false_with_num_classes(self) -> None:
        """Test BirdNET model with return_features_only=False and explicit num_classes."""
        model = BirdNetModel(return_features_only=False, num_classes=20, device="cpu")

        assert model.return_features_only is False
        assert model.num_classes == 20
        assert model.classifier is not None
        assert model.classifier.out_features == 20

    def test_return_features_only_false_without_num_classes(self) -> None:
        """Test BirdNET model with return_features_only=False and num_classes=None (defaults to 0)."""
        model = BirdNetModel(return_features_only=False, device="cpu")

        assert model.return_features_only is False
        assert model.num_classes == 0  # None defaults to 0
        assert model.classifier is None

    def test_return_features_only_overrides_num_classes(self) -> None:
        """Test that return_features_only=True overrides num_classes parameter."""
        # Even if num_classes is provided, return_features_only=True should set it to 0
        model = BirdNetModel(return_features_only=True, num_classes=50, device="cpu")

        assert model.return_features_only is True
        assert model.num_classes == 0  # Should be overridden
        assert model.classifier is None

    def test_return_features_only_forward_pass(self, audio_input: torch.Tensor) -> None:
        """Test forward pass with return_features_only=True."""
        model = BirdNetModel(return_features_only=True, device="cpu")

        with torch.no_grad():
            result = model.forward(audio_input)

        # Should return species logits (no classifier)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, model.num_species)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_return_features_only_extract_embeddings(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with return_features_only=True."""
        model = BirdNetModel(return_features_only=True, device="cpu")

        result = model.extract_embeddings(x=audio_input, aggregation="mean")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1024)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


if __name__ == "__main__":
    pytest.main([__file__])
