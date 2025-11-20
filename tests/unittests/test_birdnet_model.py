"""
Tests for the BirdNET model (TensorFlow wrapper).

Tests the new extract_embeddings interface with aggregation methods
and sequence probe compatibility.
"""

from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from representation_learning.models.birdnet import Model as BirdNetModel


class TestBirdNetModel:
    """Test suite for BirdNET model."""

    @pytest.fixture
    def mock_analyzer(self) -> Mock:
        """Mock BirdNET analyzer.

        Returns:
            Mock: A mock BirdNET analyzer with labels and interpreter.
        """
        mock_analyzer = Mock()
        mock_analyzer.labels = ["species1", "species2", "species3"]
        mock_analyzer.interpreter = Mock()
        return mock_analyzer

    @pytest.fixture
    def mock_interpreter(self) -> Mock:
        """Mock TFLite interpreter.

        Returns:
            Mock: A mock TFLite interpreter with output details and tensor methods.
        """
        mock_interpreter = Mock()
        mock_interpreter.get_output_details.return_value = [
            {"name": "embedding_output", "shape": [1, 1024]}
        ]
        mock_interpreter.get_tensor.return_value = np.random.randn(3, 1024).astype(np.float32)
        return mock_interpreter

    @pytest.fixture
    def mock_embeddings(self) -> torch.Tensor:
        """Mock embeddings tensor.

        Returns:
            torch.Tensor: A mock embeddings tensor with shape (2, 3, 1024).
        """
        return torch.randn(2, 3, 1024)  # Batch size 2, 3 chunks, embedding dim 1024

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Mock audio input tensor.

        Returns:
            torch.Tensor: A mock audio input tensor with shape (2, 144000).
        """
        return torch.randn(2, 144000)  # Batch size 2, 3 seconds at 48kHz

    @pytest.fixture
    def dict_input(self) -> Dict[str, torch.Tensor]:
        """Mock dictionary input with raw_wav.

        Returns:
            Dict[str, torch.Tensor]: A mock dictionary with raw_wav key.
        """
        return {"raw_wav": torch.randn(2, 144000)}

    @pytest.fixture
    def padding_mask(self) -> torch.Tensor:
        """Mock padding mask.

        Returns:
            torch.Tensor: A mock padding mask tensor with shape (2, 144000).
        """
        return torch.ones(2, 144000, dtype=torch.bool)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_birdnet_model_initialization(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock
    ) -> None:
        """Test BirdNET model initialization."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = Mock()

        model = BirdNetModel(num_classes=10, device="cpu")

        assert model.num_classes == 10
        assert model.num_species == 3
        assert model.classifier is not None
        assert model.classifier.in_features == 3
        assert model.classifier.out_features == 10
        assert model.device == "cpu"

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_birdnet_model_no_classifier(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock
    ) -> None:
        """Test BirdNET model without classifier."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = Mock()

        model = BirdNetModel(num_classes=0, device="cpu")

        assert model.num_classes == 0
        assert model.classifier is None

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_birdnet_model_matching_classes(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock
    ) -> None:
        """Test BirdNET model when num_classes matches num_species."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = Mock()

        model = BirdNetModel(num_classes=3, device="cpu")

        assert model.num_classes == 3
        assert model.classifier is None  # No need for additional classifier

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_aggregation_mean(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test extract_embeddings with aggregation='mean' (default)."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            result = model.extract_embeddings(x=torch.randn(2, 144000), aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1024)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_aggregation_none(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test extract_embeddings with aggregation='none' for sequence probes."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            result = model.extract_embeddings(x=torch.randn(2, 144000), aggregation="none")

            assert isinstance(result, list)
            assert len(result) == 2  # One per batch item

            # Each item should be 3D tensor (1, N, 1024)
            for item in result:
                assert isinstance(item, torch.Tensor)
                assert item.dim() == 3
                assert item.shape[0] == 1  # Batch dimension
                assert item.shape[1] == 3  # Number of chunks
                assert item.shape[2] == 1024  # Embedding dimension

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_aggregation_max(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test extract_embeddings with aggregation='max'."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            result = model.extract_embeddings(x=torch.randn(2, 144000), aggregation="max")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1024)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_aggregation_cls_token(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test extract_embeddings with aggregation='cls_token'."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            result = model.extract_embeddings(x=torch.randn(2, 144000), aggregation="cls_token")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1024)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_dict_input(
        self,
        mock_analyzer_class: Mock,
        mock_analyzer: Mock,
        mock_interpreter: Mock,
        dict_input: Dict[str, torch.Tensor],
    ) -> None:
        """Test extract_embeddings with dictionary input."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            result = model.extract_embeddings(x=dict_input, aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1024)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_invalid_aggregation(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test extract_embeddings with invalid aggregation method."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            with pytest.raises(ValueError, match="Unsupported aggregation method"):
                model.extract_embeddings(x=torch.randn(2, 144000), aggregation="invalid_method")

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_sequence_probe_compatibility(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that embeddings are compatible with sequence probes."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            # Test with aggregation="none" for sequence probes
            result = model.extract_embeddings(x=torch.randn(2, 144000), aggregation="none")

            # Should return list of 3D tensors
            assert isinstance(result, list)
            assert len(result) == 2  # One per batch item

            for _, embedding_tensor in enumerate(result):
                assert embedding_tensor.dim() == 3  # (1, N, 1024)
                assert embedding_tensor.shape == (1, 3, 1024)

                # Verify the reshaping is correct
                original_embeddings = torch.from_numpy(mock_embed.return_value)
                assert torch.allclose(embedding_tensor.squeeze(0), original_embeddings)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_device_consistency(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that embeddings are returned on the right device."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        # Mock the embedding extraction
        with patch.object(BirdNetModel, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            # Test on CPU
            model_cpu = BirdNetModel(num_classes=10, device="cpu")
            result_cpu = model_cpu.extract_embeddings(x=torch.randn(2, 144000), aggregation="mean")
            assert result_cpu.device.type == "cpu"

            # Test on CUDA if available
            if torch.cuda.is_available():
                model_cuda = BirdNetModel(num_classes=10, device="cuda")
                result_cuda = model_cuda.extract_embeddings(
                    x=torch.randn(2, 144000), aggregation="mean"
                )
                assert result_cuda.device.type == "cuda"

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_padding_mask_handling(
        self,
        mock_analyzer_class: Mock,
        mock_analyzer: Mock,
        mock_interpreter: Mock,
        padding_mask: torch.Tensor,
    ) -> None:
        """Test that padding_mask is handled correctly (even though unused)."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            # Test with padding_mask (should not affect output)
            result_with_mask = model.extract_embeddings(
                x=torch.randn(2, 144000), aggregation="mean", padding_mask=padding_mask
            )

            result_without_mask = model.extract_embeddings(
                x=torch.randn(2, 144000), aggregation="mean"
            )

            # Results should be identical since padding_mask is unused
            assert torch.allclose(result_with_mask, result_without_mask)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_extract_embeddings_consistency(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that extract_embeddings produces consistent results."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            # Test multiple calls with same input
            input_tensor = torch.randn(2, 144000)

            result1 = model.extract_embeddings(x=input_tensor, aggregation="mean")

            result2 = model.extract_embeddings(x=input_tensor, aggregation="mean")

            # Results should be identical
            assert torch.allclose(result1, result2)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_forward_method(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test the forward method with classifier."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the inference
        with patch.object(model, "_infer_clip") as mock_infer:
            mock_infer.return_value = torch.randn(3)  # Species probabilities

            result = model.forward(torch.randn(2, 144000))

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 10)  # (batch_size, num_classes)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_forward_method_no_classifier(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test the forward method without classifier."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=0, device="cpu")

        # Mock the inference
        with patch.object(model, "_infer_clip") as mock_infer:
            mock_infer.return_value = torch.randn(3)  # Species probabilities

            result = model.forward(torch.randn(2, 144000))

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 3)  # (batch_size, num_species)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_device_movement(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test device movement methods."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Test moving to CUDA
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert model_cuda.device.type == "cuda"
            if model_cuda.classifier is not None:
                assert next(model_cuda.classifier.parameters()).device.type == "cuda"

        # Test moving to CPU
        model_cpu = model.cpu()
        assert model_cpu.device.type == "cpu"
        if model_cpu.classifier is not None:
            assert next(model_cpu.classifier.parameters()).device.type == "cpu"

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_species_mapping(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test species index mapping methods."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Test idx_to_species
        assert model.idx_to_species(0) == "species1"
        assert model.idx_to_species(1) == "species2"
        assert model.idx_to_species(2) == "species3"

        # Test species_to_idx
        assert model.species_to_idx("species1") == 0
        assert model.species_to_idx("species2") == 1
        assert model.species_to_idx("species3") == 2

        # Test invalid species
        with pytest.raises(ValueError):
            model.species_to_idx("invalid_species")

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_gradient_checkpointing(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that gradient checkpointing logs a warning."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Test that gradient checkpointing logs a warning
        with patch("representation_learning.models.birdnet.logger.warning") as mock_warning:
            model.enable_gradient_checkpointing()
            mock_warning.assert_called_once_with(
                "Gradient checkpointing is not supported for BirdNET."
            )

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_model_attributes(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that model attributes are correctly set."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Test model attributes
        assert hasattr(model, "num_species")
        assert hasattr(model, "num_classes")
        assert hasattr(model, "classifier")
        assert hasattr(model, "device")

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_model_methods(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that model methods exist and are callable."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Test that methods exist
        assert hasattr(model, "extract_embeddings")
        assert hasattr(model, "forward")
        assert hasattr(model, "_embedding_for_clip")
        assert hasattr(model, "_infer_clip")
        assert hasattr(model, "idx_to_species")
        assert hasattr(model, "species_to_idx")
        assert hasattr(model, "enable_gradient_checkpointing")
        assert callable(model.extract_embeddings)
        assert callable(model.forward)
        assert callable(model._embedding_for_clip)
        assert callable(model._infer_clip)
        assert callable(model.idx_to_species)
        assert callable(model.species_to_idx)
        assert callable(model.enable_gradient_checkpointing)

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_model_initialization_edge_cases(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test model initialization with edge cases."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        # Test with very large num_classes
        model_large = BirdNetModel(num_classes=1000, device="cpu")
        assert model_large.num_classes == 1000
        assert model_large.classifier is not None
        assert model_large.classifier.out_features == 1000

    @patch("representation_learning.models.birdnet.Analyzer")
    def test_model_embedding_consistency_across_batches(
        self, mock_analyzer_class: Mock, mock_analyzer: Mock, mock_interpreter: Mock
    ) -> None:
        """Test that embeddings are consistent across different batch sizes."""
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.interpreter = mock_interpreter

        model = BirdNetModel(num_classes=10, device="cpu")

        # Mock the embedding extraction
        with patch.object(model, "_embedding_for_clip") as mock_embed:
            mock_embed.return_value = np.random.randn(3, 1024).astype(np.float32)

            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8]
            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, 144000)
                result = model.extract_embeddings(x=input_tensor, aggregation="mean")

                assert isinstance(result, torch.Tensor)
                assert result.shape == (batch_size, 1024)


if __name__ == "__main__":
    pytest.main([__file__])
