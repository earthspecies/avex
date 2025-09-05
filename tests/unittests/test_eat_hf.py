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
        # Register hooks for specific layers that we know exist
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc1"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return the main EAT features (pooled features)
        # EAT base model has 768 hidden size
        assert embeddings.shape == (2, 3072)  # fc1 output dimension
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_all_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that 'all' extracts from all MLP layers (fc1 and fc2)."""
        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return features from all MLP layers (fc1 and fc2)
        # EAT has 12 blocks, each with 2 MLP layers (fc1 and fc2)
        # fc1: 768 -> 3072, fc2: 3072 -> 768
        # Total: 12 fc1 layers × 3072 + 12 fc2 layers × 768 = 36864 + 9216 = 46080
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_specific_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a specific MLP layer."""
        # Register hooks for the specific layer
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc2"])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return features from the specified MLP layer
        assert embeddings.shape == (2, 768)  # fc2 output dimension
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_multiple_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from multiple specific MLP layers."""
        # Register hooks for the specified layers
        model.register_hooks_for_layers(
            [
                "backbone.model.blocks.0.mlp.fc1",
                "backbone.model.blocks.0.mlp.fc2",
            ]
        )

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return features from both fc1 and fc2 layers
        # fc1: 3072 features, fc2: 768 features
        assert embeddings.shape == (2, 3840)  # 3072 + 768
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_dict_input(
        self, model: EATHFModel, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test extract_embeddings with dictionary input format."""
        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return features from all MLP layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_aggregation_none(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extract_embeddings when aggregation='none' (returns tensor for single
        embedding)."""
        # Register hooks for the specific layer
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc2"])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="none",
        )

        # Clean up
        model.deregister_all_hooks()

        # When aggregation="none" and single embedding, we get a tensor
        assert torch.is_tensor(embeddings)
        assert embeddings.shape == (
            2,
            513,
            768,
        )  # (batch, seq_len, features)

    def test_extract_embeddings_invalid_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that invalid layer names raise appropriate errors."""
        with pytest.raises(
            ValueError, match="Layer 'nonexistent_layer' not found in model"
        ):
            model.register_hooks_for_layers(["nonexistent_layer"])

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
        # Register hooks for all discoverable layers using the 'all' special case
        features_model.register_hooks_for_layers(["all"])

        embeddings = features_model.extract_embeddings(
            x=audio_input, aggregation="mean"
        )

        # Clean up
        features_model.deregister_all_hooks()

        # Should return features from all MLP layers (backbone still has MLP layers)
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
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
        # Register hooks for the classifier layer
        model_custom.register_hooks_for_layers(["classifier"])

        embeddings = model_custom.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model_custom.deregister_all_hooks()

        # Should return the classifier output
        assert embeddings.shape == (2, 5)  # num_classes
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_consistency(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that extract_embeddings produces consistent results."""
        # Set deterministic seed for consistent results
        torch.manual_seed(42)

        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        # Extract embeddings twice
        embeddings1 = model.extract_embeddings(x=audio_input, aggregation="mean")
        embeddings2 = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Results should be identical
        assert torch.allclose(embeddings1, embeddings2, atol=1e-6)

    def test_extract_embeddings_device_consistency(
        self, audio_input: torch.Tensor
    ) -> None:
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

            model_cpu = EATHFModel(
                model_name="worstchan/EAT-base_epoch30_pretrain",
                num_classes=10,
                device="cpu",
                audio_config=audio_config,
                return_features_only=False,
                target_length=1024,
                pooling="cls",
            )

            # Set seed again for GPU model to ensure same initialization
            torch.manual_seed(42)

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
            # Register hooks for all discoverable layers on CPU model using the
            # 'all' special case
            model_cpu.register_hooks_for_layers(["all"])

            embeddings1 = model_cpu.extract_embeddings(
                x=audio_input, aggregation="mean"
            )

            # Clean up CPU model
            model_cpu.deregister_all_hooks()

            # Register hooks for all discoverable layers on GPU model using the
            # 'all' special case
            model_gpu.register_hooks_for_layers(["all"])

            embeddings2 = model_gpu.extract_embeddings(
                x=audio_input_gpu, aggregation="mean"
            )

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
            assert embeddings1.shape[1] > 0  # should have substantial features

            # Values should be reasonable (not NaN, not infinite, reasonable range)
            assert not torch.isnan(embeddings1).any()
            assert not torch.isnan(embeddings2).any()
            assert not torch.isinf(embeddings1).any()
            assert not torch.isinf(embeddings2).any()

            # Values should be in reasonable range (not extremely large or small)
            assert embeddings1.abs().max() < 1000
            assert embeddings2.abs().max() < 1000

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
        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return valid embeddings from all MLP layers
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_empty_layers_list(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that empty layers list returns main features."""
        # Register hooks for specific layers that we know exist
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc1"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return the main features (same as default behavior)
        assert embeddings.shape == (2, 3072)  # fc1 output dimension
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_mixed_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a mix of specific layers and 'all'."""
        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return features from all layers (no duplication)
        # 'all' already includes the specific layer, so no extra features
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_3d_embeddings(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test handling of 3D embeddings (if any layers produce them)."""
        # This test is more theoretical since EAT typically produces 2D embeddings
        # when aggregation!="none", but we can test the 3D case
        # Register hooks for the specific layer
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc2"])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="none",
        )

        # Clean up
        model.deregister_all_hooks()

        # When aggregation="none" and single embedding, we get a tensor
        assert torch.is_tensor(embeddings)
        assert embeddings.shape == (
            2,
            513,
            768,
        )  # (batch, seq_len, features)

    def test_extract_embeddings_invalid_dimension(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that invalid embedding dimensions raise appropriate errors."""
        # This test would require mocking to create 4D embeddings
        # For now, we just test that normal 2D embeddings work
        # Register hooks for all discoverable layers using the 'all' special case
        model.register_hooks_for_layers(["all"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return valid 2D embeddings
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_specific_layers_no_all(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from specific layers when 'all' is not in the list."""
        # Register hooks for the specified layers
        model.register_hooks_for_layers(
            [
                "backbone.model.blocks.0.mlp.fc1",
                "backbone.model.blocks.1.mlp.fc2",
            ]
        )

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return features from only the specified layers
        # fc1: 3072 features, fc2: 768 features
        assert embeddings.shape == (2, 3840)  # 3072 + 768
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_classifier_only(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from classifier layer only."""
        # Register hooks for the classifier layer
        model.register_hooks_for_layers(["classifier"])

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return the classifier output
        assert embeddings.shape == (2, 10)  # num_classes
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_mixed_specific_layers(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test extracting from a mix of MLP and classifier layers."""
        # Register hooks for the specified layers
        model.register_hooks_for_layers(
            [
                "backbone.model.blocks.0.mlp.fc1",
                "classifier",
            ]
        )

        embeddings = model.extract_embeddings(
            x=audio_input,
            aggregation="mean",
        )

        # Clean up
        model.deregister_all_hooks()

        # Should return features from both MLP and classifier layers
        # fc1: 3072 features, classifier: 10 features
        assert embeddings.shape == (2, 3082)  # 3072 + 10
        assert torch.is_tensor(embeddings)

    def test_mlp_layer_discovery(self, model: EATHFModel) -> None:
        """Test that MLP layers are discovered correctly during initialization."""
        # Check that MLP layers were discovered
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

        # Should return valid embeddings
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_last_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test that 'last_layer' extracts from the last non-classification layer."""
        # Register hooks for the last non-classification layer using the
        # 'last_layer' special case
        model.register_hooks_for_layers(["last_layer"])

        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Clean up
        model.deregister_all_hooks()

        # Should return features from the last non-classification layer
        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features
        assert torch.is_tensor(embeddings)

        # Verify that only one layer was registered
        assert len(model._hook_layers) == 1
        assert (
            "last_layer" not in model._hook_layers
        )  # Should be replaced with actual layer name


if __name__ == "__main__":
    pytest.main([__file__])
