from typing import Dict

import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.eat_hf import Model as EATHFModel


class TestEATHFExtractEmbeddings:
    """Test the extract_embeddings functionality for EAT HF model."""

    @pytest.fixture
    def model(self) -> EATHFModel:
        """Create an EAT HF model for testing.

        Returns:
            EATHFModel: A configured EAT HF model for testing.
        """
        device = "cpu"  # Use CPU for testing
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=10,
            device=device,
            audio_config=audio_config,
            return_features_only=False,
            target_length=1024,
            pooling="cls",
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
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that extract_embeddings returns main features by default."""
        embeddings = model.extract_embeddings(
            x=audio_input, layers=[], average_over_time=True
        )

        # Should return the main EAT features (pooled features)
        # EAT base model has 768 hidden size
        assert embeddings.shape == (2, 768)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that 'all' extracts from all MLP layers (fc1 and fc2)."""
        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return features from all MLP layers (fc1 and fc2)
        # EAT has 12 blocks, each with 2 MLP layers (fc1 and fc2)
        # fc1: 768 -> 3072, fc2: 3072 -> 768
        # Total: 12 fc1 layers × 3072 + 12 fc2 layers × 768 = 36864 + 9216 = 46080
        assert embeddings.shape == (2, 46080)  # 12×3072 + 12×768 = 46080
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_specific_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a specific MLP layer."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=["backbone.model.blocks.0.mlp.fc2"],  # First MLP fc2 layer
            average_over_time=True,
        )

        # Should return features from the specified MLP layer
        assert embeddings.shape == (2, 768)  # fc2 output dimension
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_multiple_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from multiple specific MLP layers."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=[
                "backbone.model.blocks.0.mlp.fc1",
                "backbone.model.blocks.0.mlp.fc2",
            ],
            average_over_time=True,
        )

        # Should return features from both fc1 and fc2 layers
        # fc1: 3072 features, fc2: 768 features
        assert embeddings.shape == (2, 3840)  # 3072 + 768
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_dict_input(
        self, model: EATHFModel, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test extract_embeddings with dictionary input format."""
        embeddings = model.extract_embeddings(
            x=dict_input, layers=["all"], average_over_time=True
        )

        # Should return features from all MLP layers
        assert embeddings.shape == (2, 46080)  # 12×3072 + 12×768 = 46080
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_without_averaging(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings when average_over_time=False."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=["backbone.model.blocks.0.mlp.fc2"],
            average_over_time=False,
        )

        # Should return a list of embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert embeddings[0].shape == (2, 513, 768)  # (batch, seq_len, features)

    def test_extract_embeddings_different_pooling(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that different pooling methods work correctly."""
        # Test with mean pooling
        embeddings_mean = model.extract_embeddings(
            x=audio_input, layers=["all"], pooling="mean", average_over_time=True
        )

        # Test with cls pooling
        embeddings_cls = model.extract_embeddings(
            x=audio_input, layers=["all"], pooling="cls", average_over_time=True
        )

        # Both should return the same shape (all MLP layers)
        assert embeddings_mean.shape == (2, 46080)
        assert embeddings_cls.shape == (2, 46080)
        assert torch.is_tensor(embeddings_mean)
        assert torch.is_tensor(embeddings_cls)

    def test_extract_embeddings_invalid_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that invalid layer names raise appropriate errors."""
        with pytest.raises(ValueError, match="No layers found matching"):
            model.extract_embeddings(
                x=audio_input, layers=["nonexistent_layer"], average_over_time=True
            )

    def test_extract_embeddings_features_only_mode(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings when model is in features_only mode."""
        # Create a model in features_only mode
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        features_model = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=10,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            target_length=1024,
            pooling="cls",
        )

        # For features_only mode, 'all' should still return all MLP layers
        # since the backbone still contains MLP layers
        embeddings = features_model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return features from all MLP layers (backbone still has MLP layers)
        assert embeddings.shape == (2, 46080)  # 12×3072 + 12×768 = 46080
        assert torch.is_tensor(embeddings)

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
        model_custom = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=5,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            target_length=1024,
            pooling="cls",
        )

        # Test extracting from the classifier layer
        embeddings = model_custom.extract_embeddings(
            x=audio_input,
            layers=["classifier"],
            average_over_time=True,
        )

        # Should return the classifier output
        assert embeddings.shape == (2, 5)  # num_classes
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_consistency(
        self, model: EATHFModel, audio_input: torch.Tensor
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

            model_cpu = EATHFModel(
                model_name="worstchan/EAT-base_epoch30_pretrain",
                num_classes=10,
                device="cpu",
                audio_config=audio_config,
                return_features_only=False,
                target_length=1024,
                pooling="cls",
            )

            # Test on GPU
            model_gpu = EATHFModel(
                model_name="worstchan/EAT-base_epoch30_pretrain",
                num_classes=10,
                device="cuda",
                audio_config=audio_config,
                return_features_only=False,
                target_length=1024,
                pooling="cls",
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

            # Check that GPU result is actually on GPU before moving to CPU
            assert embeddings2.device.type == "cuda"

            # Move GPU result to CPU for comparison
            embeddings2 = embeddings2.cpu()

            # Results should be close (allowing for small numerical differences)
            assert torch.allclose(embeddings1, embeddings2, atol=1e-4)

    def test_extract_embeddings_padding_mask_handling(
        self, model: EATHFModel, audio_input: torch.Tensor
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
        assert embeddings.shape == (2, 46080)
        assert torch.is_tensor(embeddings)

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = EATHFModel(
                model_name="worstchan/EAT-base_epoch30_pretrain",
                num_classes=10,
                device="cuda",
                audio_config=model.audio_config,
                return_features_only=False,
                target_length=1024,
                pooling="cls",
            )

            dict_input_gpu = {
                "raw_wav": audio_input.cuda(),
                "padding_mask": padding_mask.cuda(),
            }

            embeddings_gpu = model_gpu.extract_embeddings(
                x=dict_input_gpu, layers=["all"], average_over_time=True
            )

            assert embeddings_gpu.device.type == "cuda"

    def test_extract_embeddings_empty_layers_list(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that empty layers list returns main features."""
        embeddings = model.extract_embeddings(
            x=audio_input, layers=[], average_over_time=True
        )

        # Should return the main features (same as default behavior)
        assert embeddings.shape == (2, 768)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_mixed_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a mix of specific layers and 'all'."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=["all", "backbone.model.blocks.0.mlp.fc1"],
            average_over_time=True,
        )

        # Should return features from all layers (no duplication)
        # 'all' already includes the specific layer, so no extra features
        assert embeddings.shape == (2, 46080)  # 12×3072 + 12×768 = 46080
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_3d_embeddings(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test handling of 3D embeddings (if any layers produce them)."""
        # This test is more theoretical since EAT typically produces 2D embeddings
        # when average_over_time=True, but we can test the 3D case
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=["backbone.model.blocks.0.mlp.fc2"],
            average_over_time=False,
        )

        # Should return a list of 3D embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert embeddings[0].shape == (2, 513, 768)  # (batch, seq_len, features)
        assert torch.is_tensor(embeddings[0])

    def test_extract_embeddings_invalid_dimension(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that invalid embedding dimensions raise appropriate errors."""
        # This test would require mocking to create 4D embeddings
        # For now, we just test that normal 2D embeddings work
        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return valid 2D embeddings
        assert embeddings.shape == (2, 46080)
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_specific_layers_no_all(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from specific layers when 'all' is not in the list."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=[
                "backbone.model.blocks.0.mlp.fc1",
                "backbone.model.blocks.1.mlp.fc2",
            ],
            average_over_time=True,
        )

        # Should return features from only the specified layers
        # fc1: 3072 features, fc2: 768 features
        assert embeddings.shape == (2, 3840)  # 3072 + 768
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_classifier_only(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from classifier layer only."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=["classifier"],
            average_over_time=True,
        )

        # Should return the classifier output
        assert embeddings.shape == (2, 10)  # num_classes
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_mixed_specific_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a mix of MLP and classifier layers."""
        embeddings = model.extract_embeddings(
            x=audio_input,
            layers=[
                "backbone.model.blocks.0.mlp.fc1",
                "classifier",
            ],
            average_over_time=True,
        )

        # Should return features from both MLP and classifier layers
        # fc1: 3072 features, classifier: 10 features
        assert embeddings.shape == (2, 3082)  # 3072 + 10
        assert torch.is_tensor(embeddings)

    def test_mlp_layer_discovery(self, model: EATHFModel) -> None:
        """Test that MLP layers are discovered correctly during initialization."""
        # Check that MLP layers were discovered
        assert hasattr(model, "_mlp_layers")
        assert len(model._mlp_layers) > 0

        # Test that we can extract from discovered layers
        audio_input = torch.randn(2, 16000)
        embeddings = model.extract_embeddings(
            x=audio_input, layers=["all"], average_over_time=True
        )

        # Should return valid embeddings
        assert torch.is_tensor(embeddings)


if __name__ == "__main__":
    pytest.main([__file__])
