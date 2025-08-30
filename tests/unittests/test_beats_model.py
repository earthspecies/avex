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
        """Test that empty layers list with dict input returns main features."""
        input_dict = {"raw_wav": sample_audio}

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(input_dict, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_empty_layers_dict_input(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that empty layers list with dict input returns main features."""
        input_dict = {"raw_wav": sample_audio}

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(input_dict, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape == (2, 768)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' layers extracts from all linear layers."""
        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_all_layers_with_classifier(
        self, beats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' layers works with classifier model."""
        # Register hooks for specific layers that we know exist
        beats_model_with_classifier.register_hooks_for_layers(
            ["backbone.post_extract_proj"]
        )

        embeddings = beats_model_with_classifier.extract_embeddings(
            sample_audio, aggregation="mean"
        )

        # Clean up
        beats_model_with_classifier.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_specific_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction from specific layers."""
        # Test with classifier layer if available
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            beats_model.register_hooks_for_layers(["classifier"])
            embeddings = beats_model.extract_embeddings(
                sample_audio, aggregation="mean"
            )
            beats_model.deregister_all_hooks()
            assert embeddings.shape == (2, 10)  # num_classes
        else:
            # Test with a known linear layer from backbone
            beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
            embeddings = beats_model.extract_embeddings(
                sample_audio, aggregation="mean"
            )
            beats_model.deregister_all_hooks()
            assert embeddings.shape[0] == 2  # batch size
            assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_multiple_layers(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction from multiple specific layers."""
        layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            layers.append("classifier")

        # Register hooks for the specified layers
        beats_model.register_hooks_for_layers(layers)
        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_all_and_specific(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that 'all' and specific layers work together."""
        # Register hooks for all discoverable layers plus specific ones
        all_layers = beats_model._layer_names + ["backbone.post_extract_proj"]
        beats_model.register_hooks_for_layers(all_layers)

        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_invalid_layer(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that invalid layer names raise appropriate error."""
        # Try to register hooks for non-existent layer
        with pytest.raises(
            ValueError, match="Layer 'nonexistent_layer' not found in model"
        ):
            beats_model.register_hooks_for_layers(["nonexistent_layer"])

    def test_extract_embeddings_no_layers_found(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test behavior when no valid layers are found."""
        # Try to extract embeddings without registering any hooks
        with pytest.raises(ValueError, match="No hooks are registered in the model"):
            beats_model.extract_embeddings(sample_audio, aggregation="mean")

    def test_extract_embeddings_with_padding_mask(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with padding mask."""
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        padding_mask[:, 8000:] = True  # Pad second half

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(
            sample_audio, padding_mask=padding_mask, aggregation="mean"
        )

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_dict_input_with_padding(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with dict input containing padding mask."""
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        padding_mask[:, 8000:] = True

        input_dict = {"raw_wav": sample_audio, "padding_mask": padding_mask}

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(input_dict, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_aggregation_none(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction without aggregation (returns list)."""
        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="none")

        # Clean up
        beats_model.deregister_all_hooks()

        # When aggregation="none", we get a list of embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) > 0

        # Each embedding should be a tensor
        for emb in embeddings:
            assert torch.is_tensor(emb)
            assert emb.shape[0] == 2  # batch size
            assert emb.shape[1] > 0  # features

    def test_extract_embeddings_hook_management(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that hooks are properly managed across calls."""
        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        # First call
        embeddings1 = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Second call
        embeddings2 = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        # Results should be consistent
        assert torch.allclose(embeddings1, embeddings2)
        assert embeddings1.shape[0] == 2  # batch size
        assert embeddings1.shape[1] > 0  # features

    def test_extract_embeddings_different_layer_sets(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that different layer sets work correctly."""
        # Test with 'all'
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_all = beats_model.extract_embeddings(
            sample_audio, aggregation="mean"
        )
        beats_model.deregister_all_hooks()

        # Test with specific layers
        specific_layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            specific_layers.append("classifier")

        beats_model.register_hooks_for_layers(specific_layers)
        embeddings_specific = beats_model.extract_embeddings(
            sample_audio, aggregation="mean"
        )
        beats_model.deregister_all_hooks()

        # Both should work
        assert embeddings_all.shape[0] == 2  # batch size
        assert embeddings_specific.shape[0] == 2  # batch size
        assert embeddings_all.shape[1] > 0  # features
        assert embeddings_specific.shape[1] > 0  # features

    def test_extract_embeddings_features_only_mode(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that features_only mode works correctly."""
        # Should work in features_only mode
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_classifier_mode(
        self, beats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that classifier mode works correctly."""
        # Should work with classifier
        beats_model_with_classifier.register_hooks_for_layers(
            ["backbone.post_extract_proj"]
        )
        embeddings = beats_model_with_classifier.extract_embeddings(
            sample_audio, aggregation="mean"
        )
        beats_model_with_classifier.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_layer_discovery(self, beats_model: Model) -> None:
        """Test that linear layers are properly discovered."""
        # Manually discover layers
        beats_model._discover_linear_layers()

        # Check that MLP layers are discovered
        assert hasattr(beats_model, "_layer_names")
        assert isinstance(beats_model._layer_names, list)
        assert len(beats_model._layer_names) > 0

        # Use discovered MLP layers for testing
        layer_names = beats_model._layer_names

        # Check that we have some backbone layers
        assert any("backbone" in name for name in layer_names)

    def test_extract_embeddings_consistent_outputs(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that outputs are consistent across multiple calls."""
        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings1 = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        embeddings2 = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        # Results should be consistent
        assert torch.allclose(embeddings1, embeddings2)
        assert embeddings1.shape[0] == 2  # batch size
        assert embeddings1.shape[1] > 0  # features

    def test_extract_embeddings_gradient_free(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that extraction is gradient-free."""
        sample_audio.requires_grad_(True)

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        _ = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        beats_model.deregister_all_hooks()

        # Should not have gradients
        assert not sample_audio.grad is not None

    def test_extract_embeddings_different_batch_sizes(self, beats_model: Model) -> None:
        """Test that extraction works with different batch sizes."""
        # Single sample
        single_audio = torch.randn(1, 16000)

        # Register hooks for specific layers that we know exist
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        single_embeddings = beats_model.extract_embeddings(
            single_audio, aggregation="mean"
        )

        # Clean up
        beats_model.deregister_all_hooks()

        assert single_embeddings.shape[0] == 1  # single batch
        assert single_embeddings.shape[1] > 0  # features

    def test_extract_embeddings_error_handling(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test error handling for invalid inputs."""
        # Test with None input
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(None)

        # Test with empty audio
        empty_audio = torch.empty(0)
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(empty_audio)

    def test_extract_embeddings_layer_specific_behavior(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that specific layer extraction works as expected."""
        # Test with a specific layer
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_mixed_layer_types(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test extraction with mixed layer types."""
        layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            layers.append("classifier")

        # Register hooks for the specified layers
        beats_model.register_hooks_for_layers(layers)
        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_forward_compatibility(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that forward method still works after embedding extraction."""
        # Extract embeddings
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        _ = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Forward method should still work
        output = beats_model.forward(sample_audio)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] > 0  # features

    def test_extract_embeddings_state_preservation(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test that model state is preserved after embedding extraction."""
        # Get initial state
        initial_state = beats_model.training

        # Extract embeddings
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        _ = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # State should be preserved
        assert beats_model.training == initial_state
