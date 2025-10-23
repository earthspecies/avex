import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.beats_model import Model as BEATsModel


class TestBEATsEmbeddingExtractionIntegration:
    """Integration tests for BEATs model embedding extraction."""

    @pytest.fixture
    def beats_model(self) -> BEATsModel:
        """Create a BEATs NatureLM model for integration testing.

        Returns:
            BEATsModel: A configured BEATs model with NatureLM checkpoint for testing.
        """
        # Create audio config for BEATs
        audio_config = AudioConfig(
            sample_rate=16000,
            representation="raw",  # BEATs expects raw waveform
            normalize=False,
            target_length_seconds=10,
            window_selection="random",
        )

        # Create BEATs model with NatureLM
        model = BEATsModel(
            num_classes=10,
            pretrained=True,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,  # For embedding extraction
            use_naturelm=True,
            fine_tuned=False,
            disable_layerdrop=True,  # For consistent behavior
        )

        return model

    def test_basic_embedding_extraction(self, beats_model: BEATsModel) -> None:
        """Test basic embedding extraction functionality."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000)  # 2 seconds of audio

        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        # BEATs returns a list of tensors for different layers
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(torch.is_tensor(emb) for emb in result)
        # Check that all embeddings have the same batch size
        assert all(emb.shape[0] == 2 for emb in result)

    def test_embedding_extraction_with_dict_input(
        self, beats_model: BEATsModel
    ) -> None:
        """Test embedding extraction with dictionary input format."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = {
            "raw_wav": torch.randn(2, 16000),
            "padding_mask": torch.zeros(2, 16000, dtype=torch.bool),
        }

        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        # BEATs returns a list of tensors for different layers
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(torch.is_tensor(emb) for emb in result)
        # Check that all embeddings have the same batch size
        assert all(emb.shape[0] == 2 for emb in result)

    def test_embedding_extraction_with_padding_mask(
        self, beats_model: BEATsModel
    ) -> None:
        """Test embedding extraction with padding mask."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000)
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)

        with torch.no_grad():
            result = beats_model.extract_embeddings(x, padding_mask=padding_mask)

        # BEATs returns a list of tensors for different layers
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(torch.is_tensor(emb) for emb in result)
        # Check that all embeddings have the same batch size
        assert all(emb.shape[0] == 2 for emb in result)

    def test_mlp_layer_discovery(self, beats_model: BEATsModel) -> None:
        """Test that the model has the expected structure."""
        # BEATs model should have the expected attributes
        assert hasattr(beats_model, "backbone")
        assert hasattr(beats_model, "extract_embeddings")
        assert hasattr(beats_model, "_layer_names")

        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        # Check that hooks are registered
        assert len(beats_model._hooks) > 0
        assert len(beats_model._layer_names) > 0

    def test_embedding_extraction_all_layers_fallback(
        self, beats_model: BEATsModel
    ) -> None:
        """Test fallback behavior when no MLP layers are found."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000)

        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        # BEATs should return embeddings from all registered layers
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(torch.is_tensor(emb) for emb in result)
        assert all(emb.shape[0] == 2 for emb in result)

    def test_embedding_extraction_consistency(self, beats_model: BEATsModel) -> None:
        """Test that embedding extraction is consistent across calls."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000)

        with torch.no_grad():
            result1 = beats_model.extract_embeddings(x)
            result2 = beats_model.extract_embeddings(x)

        # Results should have same structure
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert len(result1) == len(result2)
        assert all(
            emb1.shape == emb2.shape
            for emb1, emb2 in zip(result1, result2, strict=False)
        )

    def test_embedding_extraction_different_batch_sizes(
        self, beats_model: BEATsModel
    ) -> None:
        """Test embedding extraction with different batch sizes."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 16000)

            with torch.no_grad():
                result = beats_model.extract_embeddings(x)

            # Check that all embeddings have the correct batch size
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(emb.shape[0] == batch_size for emb in result)

    def test_embedding_extraction_device_handling(
        self, beats_model: BEATsModel
    ) -> None:
        """Test that embedding extraction works on different devices."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000)

        # Test on CPU (default)
        with torch.no_grad():
            result_cpu = beats_model.extract_embeddings(x)

        # Check that all embeddings are on CPU
        assert isinstance(result_cpu, list)
        assert len(result_cpu) > 0
        assert all(emb.device == torch.device("cpu") for emb in result_cpu)
        assert all(emb.shape[0] == 2 for emb in result_cpu)

    def test_embedding_extraction_gradient_handling(
        self, beats_model: BEATsModel
    ) -> None:
        """Test that embedding extraction properly handles gradients."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        x = torch.randn(2, 16000, requires_grad=True)

        # Should work without gradients
        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        # Check that all embeddings don't require gradients
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(not emb.requires_grad for emb in result)
        assert all(emb.shape[0] == 2 for emb in result)

    def test_embedding_extraction_error_handling(self, beats_model: BEATsModel) -> None:
        """Test error handling for invalid inputs."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        # Test with empty tensor
        x = torch.tensor([])

        with pytest.raises(ValueError, match="Audio tensor cannot be empty"):
            with torch.no_grad():
                beats_model.extract_embeddings(x)

        # Test with None input
        with pytest.raises(ValueError, match="Input tensor cannot be None"):
            with torch.no_grad():
                beats_model.extract_embeddings(None)

    def test_embedding_extraction_with_realistic_audio(
        self, beats_model: BEATsModel
    ) -> None:
        """Test embedding extraction with realistic audio data."""
        # Register hooks for all layers
        beats_model.register_hooks_for_layers(["all"])

        # Create realistic audio data (sine wave)
        sample_rate = 16000
        duration = 2  # seconds
        t = torch.linspace(0, duration, sample_rate * duration)
        audio = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave

        # Add batch dimension
        x = audio.unsqueeze(0)  # (1, time)

        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        # Check that all embeddings are valid
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(emb.shape[0] == 1 for emb in result)
        assert all(torch.is_tensor(emb) for emb in result)
        assert all(not torch.isnan(emb).any() for emb in result)
        assert all(not torch.isinf(emb).any() for emb in result)
