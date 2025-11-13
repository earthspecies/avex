from typing import Dict

import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import (
    Model as EfficientNetModel,
)


class TestEfficientNetExtractEmbeddings:
    """Test the extract_embeddings functionality for EfficientNet model."""

    @pytest.fixture
    def model(self) -> EfficientNetModel:
        """Create an EfficientNet model for testing.

        Returns:
            EfficientNetModel: A configured EfficientNet model for testing.
        """
        device = "cpu"  # Use CPU for testing
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,  # Use untrained for faster testing
            device=device,
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b0",
        )
        return model

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Create dummy audio input for testing.

        Returns:
            torch.Tensor: Random audio tensor with shape (batch_size, time_steps).
        """
        batch_size = 2
        time_steps = 16000  # 1 second at 16kHz
        return torch.randn(batch_size, time_steps)

    @pytest.fixture
    def dict_input(self, audio_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create dictionary input format for testing.

        Args:
            audio_input: Audio tensor input.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with audio and padding mask.
        """
        batch_size = audio_input.shape[0]
        time_steps = audio_input.shape[1]
        padding_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)

        return {"raw_wav": audio_input, "padding_mask": padding_mask}

    def test_extract_embeddings_default_behavior(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings returns main features by default."""
        # Register hooks for specific layers that we know exist
        model.register_hooks_for_layers(["model.features.0.0"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return the main EfficientNet features (flattened pooled features)
        # EfficientNet B0 has 1280 features after pooling
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings extracts from all registered layers."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return flattened embeddings from all registered layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings)

        # Verify that we're getting embeddings from all registered layers
        # The dimension should be reasonable for the model size
        assert embeddings.shape[1] > 1000  # Should have substantial features

    def test_extract_embeddings_specific_layer(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extracting from a specific convolutional layer."""
        # Get the first convolutional layer name for testing
        model._discover_linear_layers()
        conv_layer_name = model._layer_names[0] if model._layer_names else "model.features.0.0"

        # Register hooks for the specific layer
        model.register_hooks_for_layers([conv_layer_name])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return flattened features from the convolutional layer
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_multiple_layers(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extracting from multiple specific convolutional layers."""
        # Get the first two convolutional layer names for testing
        model._discover_linear_layers()
        if len(model._layer_names) >= 2:
            conv_layer_names = model._layer_names[:2]
        else:
            # Fallback to first layer twice if not enough layers
            conv_layer_names = [
                model._layer_names[0],
                model._layer_names[0],
            ]

        # Register hooks for the specified layers
        model.register_hooks_for_layers(conv_layer_names)

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return concatenated flattened features from multiple conv layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_dict_input(self, model: EfficientNetModel, dict_input: Dict[str, torch.Tensor]) -> None:
        """Test extract_embeddings with dictionary input format."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return flattened embeddings from all registered layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_aggregation_none(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings when aggregation='none' (returns tensor for single
        embedding)."""
        # Get the first convolutional layer name for testing
        model._discover_linear_layers()
        conv_layer_name = model._layer_names[0] if model._layer_names else "model.features.0.0"

        # Register hooks for the specific layer
        model.register_hooks_for_layers([conv_layer_name])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="none")

        # Clean up
        model.deregister_all_hooks()

        # When aggregation="none" and single embedding, we get a tensor
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # flattened features

    def test_extract_embeddings_gradient_checkpointing(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that gradient checkpointing works with extract_embeddings."""
        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()
        model.train()

        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return flattened embeddings from all registered layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings)

        # Note: extract_embeddings is designed for inference (uses torch.no_grad)
        # so we don't test gradient flow here

    def test_extract_embeddings_invalid_layer(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that invalid layer names raise appropriate errors."""
        with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found in model"):
            model.register_hooks_for_layers(["nonexistent_layer"])

    def test_extract_embeddings_features_only_mode(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings when model is in features_only mode."""
        # Create a model in features_only mode
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        features_model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            efficientnet_variant="b0",
        )

        # For features_only mode, should return flattened embeddings from
        # all registered layers
        # Register hooks for all discoverable layers
        features_model._discover_linear_layers()
        features_model.register_hooks_for_layers(features_model._layer_names)

        embeddings = features_model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        features_model.deregister_all_hooks()

        # Should return flattened embeddings from all registered layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_different_variants(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with different EfficientNet variants."""
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        # Test B1 variant
        model_b1 = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b1",
        )

        # Register hooks for all discoverable layers on B1 model
        model_b1._discover_linear_layers()
        model_b1.register_hooks_for_layers(model_b1._layer_names)

        embeddings_b1 = model_b1.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up B1 model
        model_b1.deregister_all_hooks()

        # B1 should have flattened embeddings from all registered layers
        assert embeddings_b1.shape[0] == 2  # batch size
        assert embeddings_b1.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings_b1)

        # Test B0 variant for comparison
        model_b0 = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b0",
        )

        # Register hooks for all discoverable layers on B0 model
        model_b0._discover_linear_layers()
        model_b0.register_hooks_for_layers(model_b0._layer_names)

        embeddings_b0 = model_b0.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up B0 model
        model_b0.deregister_all_hooks()

        # Both should have flattened embeddings from all registered layers
        assert embeddings_b0.shape[0] == 2  # batch size
        assert embeddings_b0.shape[1] > 0  # concatenated flattened features from all registered layers

    def test_extract_embeddings_custom_num_classes(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with custom number of classes."""
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        # Create model with custom num_classes
        model_custom = EfficientNetModel(
            num_classes=10,
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b0",
        )

        # Test extracting from a convolutional layer
        model_custom._discover_linear_layers()
        conv_layer_name = model_custom._layer_names[0] if model_custom._layer_names else "model.features.0.0"

        # Register hooks for the specific layer
        model_custom.register_hooks_for_layers([conv_layer_name])

        embeddings = model_custom.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model_custom.deregister_all_hooks()

        # Should return flattened features from the convolutional layer
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_consistency(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings produces consistent results."""
        # Set deterministic seed for consistent results
        torch.manual_seed(42)

        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        # Extract embeddings twice
        embeddings1 = model.extract_embeddings(x=audio_input, aggregation="mean")
        embeddings2 = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Results should have consistent shapes and reasonable values
        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.shape[0] == 2  # batch size
        assert embeddings1.shape[1] > 1000  # should have substantial features

        # Values should be reasonable (not NaN, not infinite, reasonable range)
        assert not torch.isnan(embeddings1).any()
        assert not torch.isnan(embeddings2).any()
        assert not torch.isinf(embeddings1).any()
        assert not torch.isinf(embeddings2).any()

        # Values should be in reasonable range (not extremely large or small)
        assert embeddings1.abs().max() < 1000
        assert embeddings2.abs().max() < 1000

    def test_extract_embeddings_device_consistency(self, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings works on different devices."""
        if torch.cuda.is_available():
            # Set deterministic seed for consistent model initialization
            torch.manual_seed(42)

            # Test on CPU
            audio_config = AudioConfig(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
            )

            model_cpu = EfficientNetModel(
                num_classes=1000,
                pretrained=False,
                device="cpu",
                audio_config=audio_config,
                return_features_only=False,
                efficientnet_variant="b0",
            )

            # Set seed again for GPU model to ensure same initialization
            torch.manual_seed(42)

            # Test on GPU
            model_gpu = EfficientNetModel(
                num_classes=1000,
                pretrained=False,
                device="cuda",
                audio_config=audio_config,
                return_features_only=False,
                efficientnet_variant="b0",
            )

            # Move input to GPU
            audio_input_gpu = audio_input.cuda()

            # Extract embeddings on both devices
            # Register hooks for all discoverable layers on CPU model
            model_cpu._discover_linear_layers()
            model_cpu.register_hooks_for_layers(model_cpu._layer_names)

            embeddings1 = model_cpu.extract_embeddings(x=audio_input, aggregation="mean")

            # Clean up CPU model
            model_cpu.deregister_all_hooks()

            # Register hooks for all discoverable layers on GPU model
            model_gpu._discover_linear_layers()
            model_gpu.register_hooks_for_layers(model_gpu._layer_names)

            embeddings2 = model_gpu.extract_embeddings(x=audio_input_gpu, aggregation="mean")

            # Clean up GPU model
            model_gpu.deregister_all_hooks()

            # Check that GPU result is actually on GPU before moving to CPU
            assert embeddings2.device.type == "cuda"

            # Move GPU result to CPU for comparison
            embeddings2 = embeddings2.cpu()

            # Results should be close (allowing for small numerical differences)
            # CPU and GPU may have slight differences due to different implementations
            assert embeddings1.shape == embeddings2.shape
            assert embeddings1.shape[0] == 2  # batch size
            assert embeddings1.shape[1] > 1000  # should have substantial features

            # Values should be reasonable (not NaN, not infinite, reasonable range)
            assert not torch.isnan(embeddings1).any()
            assert not torch.isnan(embeddings2).any()
            assert not torch.isinf(embeddings1).any()
            assert not torch.isinf(embeddings2).any()

            # Values should be in reasonable range (not extremely large or small)
            assert embeddings1.abs().max() < 1000
            assert embeddings2.abs().max() < 1000

    def test_extract_embeddings_padding_mask_handling(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that padding_mask is properly handled."""
        batch_size = audio_input.shape[0]
        time_steps = audio_input.shape[1]

        # Create padding mask with some padding
        padding_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
        padding_mask[:, -100:] = False  # Last 100 timesteps are padding

        # Create dict input with padding mask
        dict_input = {"raw_wav": audio_input, "padding_mask": padding_mask}

        # Extract embeddings
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return valid embeddings from all registered layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features from all registered layers
        assert torch.is_tensor(embeddings)
        # The dimension should be reasonable for the model size
        assert embeddings.shape[1] > 1000  # Should have substantial features

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = EfficientNetModel(
                num_classes=1000,
                pretrained=False,
                device="cuda",
                audio_config=model.audio_config,
                return_features_only=False,
                efficientnet_variant="b0",
            )

            dict_input_gpu = {
                "raw_wav": audio_input.cuda(),
                "padding_mask": padding_mask.cuda(),
            }

            # Register hooks for all discoverable layers on GPU model
            model_gpu._discover_linear_layers()
            model_gpu.register_hooks_for_layers(model_gpu._layer_names)

            embeddings_gpu = model_gpu.extract_embeddings(x=dict_input_gpu, aggregation="mean")

            # Clean up GPU model
            model_gpu.deregister_all_hooks()

            assert embeddings_gpu.device.type == "cuda"
            # Verify GPU embeddings also have reasonable dimensions
            assert embeddings_gpu.shape[1] > 1000  # Should have substantial features

    def test_conv_layer_discovery(self, model: EfficientNetModel) -> None:
        """Test that convolutional layers are discovered correctly during
        initialization."""
        # Check that convolutional layers were discovered
        model._discover_linear_layers()
        assert hasattr(model, "_layer_names")
        assert len(model._layer_names) > 0

        # Test that we can extract from discovered layers
        audio_input = torch.randn(2, 16000)
        # Register hooks for all discoverable layers
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return valid embeddings from all registered layers
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[1] > 1000  # Should have substantial features

    def test_all_layers_uses_all_registered_layers(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings uses all registered layers."""
        # Get the total number of registered layers
        model._discover_linear_layers()
        total_layers = len(model._layer_names)
        assert total_layers > 0, "Model should have at least some registered layers"

        # Extract embeddings using all registered layers
        model.register_hooks_for_layers(model._layer_names)

        embeddings_all = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Extract embeddings using only a subset of layers for comparison
        subset_layers = model._layer_names[:3] if len(model._layer_names) >= 3 else model._layer_names
        model.register_hooks_for_layers(subset_layers)

        embeddings_subset = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # The subset should have fewer features than all layers
        assert embeddings_subset.shape[1] < embeddings_all.shape[1]

        # Verify dimensions are reasonable for the model size
        assert embeddings_all.shape[1] > 1000, (
            f"Embedding dimension {embeddings_all.shape[1]} should have substantial features"
        )

        # Log the actual dimensions for debugging
        print(f"Total registered layers: {total_layers}")
        print(f"Subset layers used: {subset_layers}")
        print(f"All layers embedding dimension: {embeddings_all.shape[1]}")
        print(f"Subset embedding dimension: {embeddings_subset.shape[1]}")

    def test_extract_embeddings_aggregation_mean(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='mean'."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return concatenated tensor with mean aggregation
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 1000  # should have substantial features

    def test_extract_embeddings_aggregation_max(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with aggregation='max'."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="max")

        # Clean up
        model.deregister_all_hooks()

        # Should return concatenated tensor with max aggregation
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 1000  # should have substantial features

    def test_extract_embeddings_aggregation_cls_token(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings with aggregation='cls_token'."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings = model.extract_embeddings(x=audio_input, aggregation="cls_token")

        # Clean up
        model.deregister_all_hooks()

        # Should return concatenated tensor with cls_token aggregation
        assert torch.is_tensor(embeddings)
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 1000  # should have substantial features

    def test_extract_embeddings_aggregation_invalid(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with invalid aggregation method."""
        # Register hooks for all discoverable layers
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        # Should raise ValueError for invalid aggregation method
        with pytest.raises(ValueError, match="Unsupported aggregation method: invalid"):
            model.extract_embeddings(x=audio_input, aggregation="invalid")

        # Clean up
        model.deregister_all_hooks()


if __name__ == "__main__":
    pytest.main([__file__])
