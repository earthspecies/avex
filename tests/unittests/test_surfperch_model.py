"""
Tests for the SurfPerch model (TensorFlow wrapper).

Tests the new extract_embeddings interface with aggregation methods
and sequence probe compatibility.
"""

from typing import Dict
from unittest.mock import Mock, patch

import pytest
import torch

from representation_learning.models.surfperch import PerchModel


class TestSurfPerchModel:
    """Test suite for SurfPerchModel."""

    @pytest.fixture
    def mock_tf_model(self) -> Mock:
        """Mock TensorFlow model for testing.

        Returns:
            Mock: A mock TensorFlow model with serving signatures.
        """
        mock_model = Mock()
        mock_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        return mock_model

    @pytest.fixture
    def mock_embeddings(self) -> torch.Tensor:
        """Mock embeddings tensor.

        Returns:
            torch.Tensor: A mock embeddings tensor with shape (2, 1280).
        """
        return torch.randn(2, 1280)  # Batch size 2, embedding dim 1280

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Mock audio input tensor.

        Returns:
            torch.Tensor: A mock audio input tensor with shape (2, 160000).
        """
        return torch.randn(2, 160000)  # Batch size 2, 5 seconds at 32kHz

    @pytest.fixture
    def dict_input(self) -> Dict[str, torch.Tensor]:
        """Mock dictionary input with raw_wav.

        Returns:
            Dict[str, torch.Tensor]: A mock dictionary with raw_wav key.
        """
        return {"raw_wav": torch.randn(2, 160000)}

    @pytest.fixture
    def padding_mask(self) -> torch.Tensor:
        """Mock padding mask.

        Returns:
            torch.Tensor: A mock padding mask tensor with shape (2, 160000).
        """
        return torch.ones(2, 160000, dtype=torch.bool)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_surfperch_model_initialization(self, mock_load_tf: Mock) -> None:
        """Test SurfPerchModel initialization."""
        # Mock the TF model and its output
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        # Mock the TF forward pass to return embeddings
        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            assert model.num_classes == 10
            assert model.target_sr == 32000
            assert model.window_samples == 160000
            assert model.embedding_dim == 1280
            assert model.classifier is not None
            assert model.classifier.in_features == 1280
            assert model.classifier.out_features == 10

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_surfperch_model_no_classifier(self, mock_load_tf: Mock) -> None:
        """Test SurfPerchModel without classifier."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=0, device="cpu")

            assert model.num_classes == 0
            assert model.classifier is None

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_aggregation_mean(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='mean' (default)."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test with tensor input
            result = model.extract_embeddings(x=torch.randn(2, 160000), aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1280)
            assert torch.allclose(result, mock_embeddings)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_aggregation_none(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='none' for sequence probes."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test with aggregation="none"
            result = model.extract_embeddings(x=torch.randn(2, 160000), aggregation="none")

            assert isinstance(result, list)
            assert len(result) == 1  # Single layer model
            assert isinstance(result[0], torch.Tensor)
            assert result[0].shape == (2, 1, 1280)  # (B, 1, embed_dim)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_aggregation_max(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='max'."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            result = model.extract_embeddings(x=torch.randn(2, 160000), aggregation="max")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1280)
            assert torch.allclose(result, mock_embeddings)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_aggregation_cls_token(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='cls_token'."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            result = model.extract_embeddings(x=torch.randn(2, 160000), aggregation="cls_token")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1280)
            assert torch.allclose(result, mock_embeddings)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_dict_input(
        self,
        mock_load_tf: Mock,
        mock_embeddings: torch.Tensor,
        dict_input: Dict[str, torch.Tensor],
    ) -> None:
        """Test extract_embeddings with dictionary input."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            result = model.extract_embeddings(x=dict_input, aggregation="mean")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1280)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_invalid_aggregation(self, mock_load_tf: Mock) -> None:
        """Test extract_embeddings with invalid aggregation method."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            with pytest.raises(ValueError, match="Unsupported aggregation method"):
                model.extract_embeddings(x=torch.randn(2, 160000), aggregation="invalid_method")

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_sequence_probe_compatibility(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test that embeddings are compatible with sequence probes."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test with aggregation="none" for sequence probes
            result = model.extract_embeddings(x=torch.randn(2, 160000), aggregation="none")

            # Should return list of 3D tensors
            assert isinstance(result, list)
            assert len(result) == 1

            embedding_tensor = result[0]
            assert embedding_tensor.dim() == 3  # (B, 1, embed_dim)
            assert embedding_tensor.shape == (2, 1, 1280)

            # Verify the reshaping is correct
            assert torch.allclose(embedding_tensor.squeeze(1), mock_embeddings)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_device_consistency(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test that embeddings are returned on the correct device."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            # Test on CPU
            model_cpu = PerchModel(num_classes=10, device="cpu")
            result_cpu = model_cpu.extract_embeddings(x=torch.randn(2, 160000), aggregation="mean")
            assert result_cpu.device.type == "cpu"

            # Test on CUDA if available
            if torch.cuda.is_available():
                model_cuda = PerchModel(num_classes=10, device="cuda")
                result_cuda = model_cuda.extract_embeddings(
                    x=torch.randn(2, 160000), aggregation="mean"
                )
                assert result_cuda.device.type == "cuda"

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_padding_mask_handling(
        self,
        mock_load_tf: Mock,
        mock_embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        """Test that padding_mask is handled correctly (even though unused)."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test with padding_mask (should not affect output)
            result_with_mask = model.extract_embeddings(
                x=torch.randn(2, 160000), aggregation="mean", padding_mask=padding_mask
            )

            result_without_mask = model.extract_embeddings(
                x=torch.randn(2, 160000), aggregation="mean"
            )

            # Results should be identical since padding_mask is unused
            assert torch.allclose(result_with_mask, result_without_mask)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_extract_embeddings_consistency(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test that extract_embeddings produces consistent results."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test multiple calls with same input
            input_tensor = torch.randn(2, 160000)

            result1 = model.extract_embeddings(x=input_tensor, aggregation="mean")

            result2 = model.extract_embeddings(x=input_tensor, aggregation="mean")

            # Results should be identical
            assert torch.allclose(result1, result2)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_forward_method(self, mock_load_tf: Mock, mock_embeddings: torch.Tensor) -> None:
        """Test the forward method with classifier."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=10, device="cpu")

            # Test forward pass
            result = model.forward(torch.randn(2, 160000))

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 10)  # (batch_size, num_classes)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_forward_method_no_classifier(
        self, mock_load_tf: Mock, mock_embeddings: torch.Tensor
    ) -> None:
        """Test the forward method without classifier."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = mock_embeddings

            model = PerchModel(num_classes=0, device="cpu")

            # Test forward pass (should return features)
            result = model.forward(torch.randn(2, 160000))

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 1280)  # (batch_size, embedding_dim)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_model_attributes(self, mock_load_tf: Mock) -> None:
        """Test that model attributes are correctly set."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            # Test model attributes
            assert hasattr(model, "target_sr")
            assert hasattr(model, "window_samples")
            assert hasattr(model, "embedding_dim")
            assert hasattr(model, "num_classes")
            assert hasattr(model, "classifier")
            assert hasattr(model, "device")

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_model_methods(self, mock_load_tf: Mock) -> None:
        """Test that model methods exist and are callable."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            # Test that methods exist
            assert hasattr(model, "extract_embeddings")
            assert hasattr(model, "forward")
            assert callable(model.extract_embeddings)
            assert callable(model.forward)

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_model_device_handling(self, mock_load_tf: Mock) -> None:
        """Test model device handling methods."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            # Test device movement methods
            if torch.cuda.is_available():
                model_cuda = model.cuda()
                # The device attribute remains the same, but parameters are moved
                assert model_cuda.device == "cpu"  # String attribute doesn't change
                if model_cuda.classifier is not None:
                    assert next(model_cuda.classifier.parameters()).device.type == "cuda"

            model_cpu = model.cpu()
            assert model_cpu.device == "cpu"  # String attribute doesn't change
            if model_cpu.classifier is not None:
                assert next(model_cpu.classifier.parameters()).device.type == "cpu"

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_model_initialization_edge_cases(self, mock_load_tf: Mock) -> None:
        """Test model initialization with edge cases."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            # Test with very large num_classes
            model_large = PerchModel(num_classes=1000, device="cpu")
            assert model_large.num_classes == 1000
            assert model_large.classifier is not None
            assert model_large.classifier.out_features == 1000

    @patch("representation_learning.models.surfperch._load_tf_model")
    def test_model_embedding_consistency_across_batches(self, mock_load_tf: Mock) -> None:
        """Test that embeddings are consistent across different batch sizes."""
        mock_tf_model = Mock()
        mock_tf_model.signatures = {"serving_default": Mock(return_value={"output_1": Mock()})}
        mock_load_tf.return_value = mock_tf_model

        with patch("representation_learning.models.surfperch.torch.from_numpy") as mock_from_numpy:
            mock_from_numpy.return_value = torch.randn(1, 1280)

            model = PerchModel(num_classes=10, device="cpu")

            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8]
            for batch_size in batch_sizes:
                # Mock the embedding extraction for this specific batch size
                with patch.object(model, "extract_features") as mock_extract:
                    mock_extract.return_value = torch.randn(batch_size, 1280)

                    input_tensor = torch.randn(batch_size, 160000)
                    result = model.extract_embeddings(x=input_tensor, aggregation="mean")

                    assert isinstance(result, torch.Tensor)
                    assert result.shape == (batch_size, 1280)


if __name__ == "__main__":
    pytest.main([__file__])
