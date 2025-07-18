"""Tests for BEATs model embedding extraction functionality."""

import pytest
import torch

from representation_learning.models.beats_model import Model


class TestBEATsModelEmbeddingExtraction:
    """Test BEATs model embedding extraction functionality."""

    @pytest.fixture
    def beats_model(self) -> Model:
        """Create a BEATs model for testing.

        Returns:
            Model: A configured BEATs model for testing.
        """
        return Model(num_classes=10, return_features_only=True, device="cpu")

    @pytest.fixture
    def beats_model_with_classifier(self) -> Model:
        """Create a BEATs model with classifier for testing.

        Returns:
            Model: A configured BEATs model with classifier for testing.
        """
        return Model(num_classes=10, return_features_only=False, device="cpu")

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor.

        Returns:
            torch.Tensor: Random audio tensor with shape (batch_size, time_steps).
        """
        return torch.randn(2, 16000)  # 2 seconds at 16kHz

    def test_extract_embeddings_empty_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that empty layers list returns main features."""
        embeddings = beats_model.extract_embeddings(sample_audio, [])

        assert embeddings.shape == (2, 768)  # BEATs default embedding dim
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_empty_layers_dict_input(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that empty layers list with dict input returns main features."""
        input_dict = {"raw_wav": sample_audio}
        embeddings = beats_model.extract_embeddings(input_dict, [])

        assert embeddings.shape == (2, 768)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' layers extracts from all linear layers."""
        embeddings = beats_model.extract_embeddings(sample_audio, ["all"])

        # Should have embeddings from all linear layers
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2  # batch size

        # Check that we have a reasonable number of features
        # BEATs has many linear layers, so total features should be large
        assert embeddings.shape[1] > 768

    def test_extract_embeddings_all_layers_with_classifier(
        self, beats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' layers works with classifier model."""
        embeddings = beats_model_with_classifier.extract_embeddings(
            sample_audio, ["all"]
        )

        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 768

    def test_extract_embeddings_specific_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction from specific layers."""
        # Test with classifier layer if available
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            embeddings = beats_model.extract_embeddings(sample_audio, ["classifier"])
            assert embeddings.shape == (2, 10)  # num_classes
        else:
            # Test with a known linear layer from backbone
            embeddings = beats_model.extract_embeddings(
                sample_audio, ["backbone.post_extract_proj"]
            )
            assert torch.is_tensor(embeddings)
            assert embeddings.shape[0] == 2

    def test_extract_embeddings_multiple_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction from multiple specific layers."""
        layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            layers.append("classifier")

        embeddings = beats_model.extract_embeddings(sample_audio, layers)
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2

    def test_extract_embeddings_all_and_specific(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' and specific layers work together."""
        layers = ["all", "backbone.post_extract_proj"]
        embeddings = beats_model.extract_embeddings(sample_audio, layers)

        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 768

    def test_extract_embeddings_invalid_layer(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that invalid layer names raise appropriate error."""
        with pytest.raises(ValueError, match="No layers found matching"):
            beats_model.extract_embeddings(sample_audio, ["nonexistent_layer"])

    def test_extract_embeddings_no_layers_found(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test behavior when no valid layers are found."""
        with pytest.raises(ValueError, match="No layers found matching"):
            beats_model.extract_embeddings(sample_audio, ["invalid1", "invalid2"])

    def test_extract_embeddings_with_padding_mask(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with padding mask."""
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        padding_mask[:, 8000:] = True  # Pad second half

        embeddings = beats_model.extract_embeddings(
            sample_audio, ["all"], padding_mask=padding_mask
        )

        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2

    def test_extract_embeddings_dict_input_with_padding(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with dict input containing padding mask."""
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        padding_mask[:, 8000:] = True

        input_dict = {"raw_wav": sample_audio, "padding_mask": padding_mask}
        embeddings = beats_model.extract_embeddings(input_dict, ["all"])

        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2

    def test_extract_embeddings_average_over_time_false(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction without averaging over time."""
        embeddings = beats_model.extract_embeddings(
            sample_audio, ["all"], average_over_time=False
        )

        # Should return list of embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) > 0

        # Each embedding should be 3D or 4D (batch, time, features) or
        # (batch, heads, time, features)
        for emb in embeddings:
            assert emb.dim() in [3, 4]
            assert emb.shape[0] == 2

    def test_extract_embeddings_hook_management(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that hooks are properly managed across calls."""
        # First call
        embeddings1 = beats_model.extract_embeddings(sample_audio, ["all"])

        # Second call - should work without interference
        embeddings2 = beats_model.extract_embeddings(sample_audio, ["all"])

        assert torch.is_tensor(embeddings1)
        assert torch.is_tensor(embeddings2)
        assert embeddings1.shape == embeddings2.shape

    def test_extract_embeddings_different_layer_sets(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that different layer sets work correctly."""
        # Test with 'all'
        embeddings_all = beats_model.extract_embeddings(sample_audio, ["all"])

        # Test with specific layers
        embeddings_specific = beats_model.extract_embeddings(
            sample_audio, ["backbone.post_extract_proj"]
        )

        assert torch.is_tensor(embeddings_all)
        assert torch.is_tensor(embeddings_specific)
        assert (
            embeddings_all.shape[0] == embeddings_specific.shape[0]
        )  # Same batch size

    def test_extract_embeddings_features_only_mode(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that features_only mode works correctly."""
        # Should work in features_only mode
        embeddings = beats_model.extract_embeddings(sample_audio, ["all"])
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_classifier_mode(
        self, beats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that classifier mode works correctly."""
        # Should work with classifier
        embeddings = beats_model_with_classifier.extract_embeddings(
            sample_audio, ["all"]
        )
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_layer_discovery(self, beats_model: Model) -> None:
        """Test that linear layers are properly discovered."""
        assert hasattr(beats_model, "_linear_layer_names")
        assert isinstance(beats_model._linear_layer_names, list)
        assert len(beats_model._linear_layer_names) > 0

        # Check that we have some backbone layers
        layer_names = beats_model._linear_layer_names
        assert any("backbone" in name for name in layer_names)

    def test_extract_embeddings_consistent_outputs(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that outputs are consistent across multiple calls."""
        embeddings1 = beats_model.extract_embeddings(sample_audio, ["all"])
        embeddings2 = beats_model.extract_embeddings(sample_audio, ["all"])

        assert torch.is_tensor(embeddings1)
        assert torch.is_tensor(embeddings2)
        torch.testing.assert_close(embeddings1, embeddings2)

    def test_extract_embeddings_gradient_free(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that extraction is gradient-free."""
        sample_audio.requires_grad_(True)

        _ = beats_model.extract_embeddings(sample_audio, ["all"])

        # Should not compute gradients
        assert not sample_audio.grad is not None

    def test_extract_embeddings_different_batch_sizes(self, beats_model: Model) -> None:
        """Test that extraction works with different batch sizes."""
        # Single sample
        single_audio = torch.randn(1, 16000)
        single_embeddings = beats_model.extract_embeddings(single_audio, ["all"])
        assert single_embeddings.shape[0] == 1

        # Multiple samples
        multi_audio = torch.randn(4, 16000)
        multi_embeddings = beats_model.extract_embeddings(multi_audio, ["all"])
        assert multi_embeddings.shape[0] == 4

    def test_extract_embeddings_error_handling(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test error handling for invalid inputs."""
        # Test with None input
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(None, ["all"])

        # Test with empty audio
        empty_audio = torch.randn(0, 16000)
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(empty_audio, ["all"])

    def test_extract_embeddings_layer_specific_behavior(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that specific layer extraction works as expected."""
        # Test with a specific layer
        embeddings = beats_model.extract_embeddings(
            sample_audio, ["backbone.post_extract_proj"]
        )
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2

        # Test with empty audio
        empty_audio = torch.randn(2, 0)
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(empty_audio, ["all"])

    def test_extract_embeddings_mixed_layer_types(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with mixed layer types."""
        layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            layers.append("classifier")

        embeddings = beats_model.extract_embeddings(sample_audio, layers)
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_extract_embeddings_forward_compatibility(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that forward method still works after embedding extraction."""
        # Extract embeddings
        _ = beats_model.extract_embeddings(sample_audio, ["all"])

        # Forward should still work
        output = beats_model(sample_audio)
        assert torch.is_tensor(output)
        assert output.shape[0] == 2

    def test_extract_embeddings_state_preservation(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that model state is preserved after embedding extraction."""
        # Get initial state
        initial_state = beats_model.training

        # Extract embeddings
        _ = beats_model.extract_embeddings(sample_audio, ["all"])

        # State should be preserved
        assert beats_model.training == initial_state
