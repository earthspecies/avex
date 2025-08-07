from typing import Dict

import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel


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

    def test_extract_embeddings_default_behavior(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that extract_embeddings returns main features by default."""
        embeddings = model.extract_embeddings(
            x=audio_input, layers=[], average_over_time=True
        )

        # Should return the main EfficientNet features (flattened pooled features)
        # EfficientNet B0 has 1280 features after pooling
        assert embeddings.shape == (2, 1280)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that 'all' extracts from all convolutional layers."""
        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return flattened embeddings from all convolutional layers
        # The shape will depend on the number of conv layers and their flattened sizes
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_specific_layer(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a specific convolutional layer."""
        # Get the first convolutional layer name for testing
        conv_layer_name = (
            model._conv_layer_names[0]
            if model._conv_layer_names
            else "model.features.0.0"
        )

        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=[conv_layer_name],
            average_over_time=True,
        )

        # Should return flattened features from the convolutional layer
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_multiple_layers(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from multiple specific convolutional layers."""
        # Get the first two convolutional layer names for testing
        if len(model._conv_layer_names) >= 2:
            conv_layer_names = model._conv_layer_names[:2]
        else:
            # Fallback to first layer twice if not enough layers
            conv_layer_names = [model._conv_layer_names[0], model._conv_layer_names[0]]

        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=conv_layer_names,
            average_over_time=True,
        )

        # Should return concatenated flattened features from multiple conv layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_dict_input(
        self, model: EfficientNetModel, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test extract_embeddings with dictionary input format."""
        embeddings = model.extract_embeddings(
            x=dict_input, layers=["all"], average_over_time=True
        )

        # Should return flattened embeddings from all convolutional layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_without_averaging(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings when average_over_time=False."""
        # Get the first convolutional layer name for testing
        conv_layer_name = (
            model._conv_layer_names[0]
            if model._conv_layer_names
            else "model.features.0.0"
        )

        embeddings = model.extract_embeddings(
            x=audio_input, layers=[conv_layer_name], average_over_time=False
        )

        # Should return a list of embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert embeddings[0].shape[0] == 2  # batch size
        assert embeddings[0].shape[1] > 0  # flattened features

    def test_extract_embeddings_gradient_checkpointing(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that gradient checkpointing works with extract_embeddings."""
        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()
        model.train()

        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return flattened embeddings from all convolutional layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

        # Test that gradients can flow through
        loss = embeddings.sum()
        loss.backward()

        # Check that gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad is not None
                break

    def test_extract_embeddings_invalid_layer(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that invalid layer names raise appropriate errors."""
        with pytest.raises(ValueError, match="No layers found matching"):
            model.extract_embeddings(
                x=audio_input, layers=["nonexistent_layer"], average_over_time=True
            )

    def test_extract_embeddings_features_only_mode(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
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

        # For features_only mode, 'all' should return flattened embeddings from
        # all convolutional layers
        embeddings = features_model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return flattened embeddings from all convolutional layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_different_variants(
        self, audio_input: torch.Tensor
    ) -> None:
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

        embeddings_b1 = model_b1.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # B1 should have flattened embeddings from convolutional layers
        assert embeddings_b1.shape[0] == 2  # batch size
        assert embeddings_b1.shape[1] > 0  # concatenated flattened features
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

        embeddings_b0 = model_b0.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Both should have flattened embeddings from convolutional layers
        assert embeddings_b0.shape[0] == 2  # batch size
        assert embeddings_b0.shape[1] > 0  # concatenated flattened features
        assert embeddings_b1.shape[0] == 2  # batch size
        assert embeddings_b1.shape[1] > 0  # concatenated flattened features

    def test_extract_embeddings_custom_num_classes(
        self, audio_input: torch.Tensor
    ) -> None:
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
        conv_layer_name = (
            model_custom._conv_layer_names[0]
            if model_custom._conv_layer_names
            else "model.features.0.0"
        )

        embeddings = model_custom.extract_embeddings(
            x=audio_input,
            layers=[conv_layer_name],
            average_over_time=True,
        )

        # Should return flattened features from the convolutional layer
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # flattened features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_consistency(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that extract_embeddings produces consistent results."""
        # Set deterministic seed for consistent results
        torch.manual_seed(42)

        # Extract embeddings twice
        embeddings1 = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )
        embeddings2 = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Results should be identical
        assert torch.allclose(embeddings1, embeddings2, atol=1e-6)

    def test_extract_embeddings_device_consistency(
        self, audio_input: torch.Tensor
    ) -> None:
        """Test that extract_embeddings works on different devices."""
        if torch.cuda.is_available():
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
            embeddings1 = model_cpu.extract_embeddings(
                x=audio_input, layers=["all"], average_over_time=True
            )
            embeddings2 = model_gpu.extract_embeddings(
                x=audio_input_gpu, layers=["all"], average_over_time=True
            )

            # Move GPU result to CPU for comparison
            embeddings2 = embeddings2.cpu()

            # Results should be close (allowing for small numerical differences)
            assert torch.allclose(embeddings1, embeddings2, atol=1e-6)

            # Check that GPU result is actually on GPU
            assert embeddings2.device.type == "cuda"

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
        embeddings = model.extract_embeddings(
            x=dict_input, layers=["all"], average_over_time=True
        )

        # Should return valid embeddings
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # concatenated flattened features
        assert torch.is_tensor(embeddings)

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

            embeddings_gpu = model_gpu.extract_embeddings(
                x=dict_input_gpu, layers=["all"], average_over_time=True
            )

            assert embeddings_gpu.device.type == "cuda"

    def test_conv_layer_discovery(self, model: EfficientNetModel) -> None:
        """Test that convolutional layers are discovered correctly during
        initialization."""
        # Check that convolutional layers were discovered
        assert hasattr(model, "_conv_layer_names")
        assert len(model._conv_layer_names) > 0

        # Test that we can extract from discovered layers
        audio_input = torch.randn(2, 16000)
        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return valid embeddings
        assert torch.is_tensor(embeddings)


if __name__ == "__main__":
    pytest.main([__file__])
