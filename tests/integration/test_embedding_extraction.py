import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.beats_model import Model as BEATsModel


class TestBEATsEmbeddingExtractionIntegration:
    """Integration tests for BEATs model embedding extraction."""

    @pytest.fixture(scope="class")
    def beats_model(self) -> BEATsModel:
        """Create a BEATs NatureLM model for integration testing.

        Model is loaded once per test class to improve performance.

        Returns:
            BEATsModel: A configured BEATs model with NatureLM checkpoint for testing.
        """
        audio_config = AudioConfig(
            sample_rate=16000,
            representation="raw",
            normalize=False,
            target_length_seconds=10,
            window_selection="random",
        )

        model = BEATsModel(
            num_classes=10,
            pretrained=True,
            device="cpu",
            audio_config=audio_config,
            return_features_only=True,
            use_naturelm=True,
            fine_tuned=False,
            disable_layerdrop=True,
        )

        return model

    @pytest.fixture(autouse=True)
    def cleanup_hooks(self, request: pytest.FixtureRequest) -> None:
        """Ensure hooks are cleaned up after each test.

        Args:
            request: Pytest request object to access test fixtures.

        Yields:
            None: Yields control to the test, then cleans up hooks after.
        """
        yield
        if "beats_model" in request.fixturenames:
            beats_model = request.getfixturevalue("beats_model")
            beats_model.deregister_all_hooks()

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor.

        Returns:
            torch.Tensor: Random audio tensor with shape (2, 16000).
        """
        return torch.randn(2, 16000)

    def test_embedding_extraction_input_formats(self, beats_model: BEATsModel, sample_audio: torch.Tensor) -> None:
        """Test embedding extraction with tensor, dict, and padding mask inputs."""
        beats_model.register_hooks_for_layers(["all"])

        # Tensor input
        with torch.no_grad():
            result_tensor = beats_model.extract_embeddings(sample_audio)

        # Dict input
        dict_input = {
            "raw_wav": torch.randn(2, 16000),
            "padding_mask": torch.zeros(2, 16000, dtype=torch.bool),
        }
        with torch.no_grad():
            result_dict = beats_model.extract_embeddings(dict_input)

        # Padding mask parameter
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        with torch.no_grad():
            result_padding = beats_model.extract_embeddings(sample_audio, padding_mask=padding_mask)

        # All should work
        for result in [result_tensor, result_dict, result_padding]:
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(torch.is_tensor(emb) for emb in result)
            assert all(emb.shape[0] == 2 for emb in result)

    def test_embedding_extraction_consistency_and_structure(
        self, beats_model: BEATsModel, sample_audio: torch.Tensor
    ) -> None:
        """Test embedding extraction consistency and model structure."""
        # Check model structure
        assert hasattr(beats_model, "backbone")
        assert hasattr(beats_model, "extract_embeddings")
        assert hasattr(beats_model, "_layer_names")

        beats_model.register_hooks_for_layers(["all"])

        # Check hooks are registered
        assert len(beats_model._hooks) > 0
        assert len(beats_model._layer_names) > 0

        # Test consistency
        with torch.no_grad():
            result1 = beats_model.extract_embeddings(sample_audio)
            result2 = beats_model.extract_embeddings(sample_audio)

        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert len(result1) == len(result2)
        assert all(emb1.shape == emb2.shape for emb1, emb2 in zip(result1, result2, strict=False))

    def test_embedding_extraction_batch_sizes(self, beats_model: BEATsModel) -> None:
        """Test embedding extraction with different batch sizes."""
        beats_model.register_hooks_for_layers(["all"])

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 16000)
            with torch.no_grad():
                result = beats_model.extract_embeddings(x)

            assert isinstance(result, list)
            assert len(result) > 0
            assert all(emb.shape[0] == batch_size for emb in result)

    def test_embedding_extraction_device_and_gradient_handling(
        self, beats_model: BEATsModel, sample_audio: torch.Tensor
    ) -> None:
        """Test device handling and gradient behavior."""
        beats_model.register_hooks_for_layers(["all"])

        # Device handling
        with torch.no_grad():
            result_cpu = beats_model.extract_embeddings(sample_audio)

        assert isinstance(result_cpu, list)
        assert all(emb.device == torch.device("cpu") for emb in result_cpu)

        # Gradient handling
        x_grad = torch.randn(2, 16000, requires_grad=True)
        with torch.no_grad():
            result_grad = beats_model.extract_embeddings(x_grad)

        assert isinstance(result_grad, list)
        assert all(not emb.requires_grad for emb in result_grad)
        assert all(emb.shape[0] == 2 for emb in result_grad)

    def test_embedding_extraction_error_handling(self, beats_model: BEATsModel) -> None:
        """Test error handling for invalid inputs."""
        beats_model.register_hooks_for_layers(["all"])

        # Empty tensor
        x_empty = torch.tensor([])
        with pytest.raises(ValueError, match="Audio tensor cannot be empty"):
            with torch.no_grad():
                beats_model.extract_embeddings(x_empty)

        # None input
        with pytest.raises(ValueError, match="Input tensor cannot be None"):
            with torch.no_grad():
                beats_model.extract_embeddings(None)

    def test_embedding_extraction_realistic_audio(self, beats_model: BEATsModel) -> None:
        """Test embedding extraction with realistic audio data."""
        beats_model.register_hooks_for_layers(["all"])

        # Create realistic audio (sine wave)
        sample_rate = 16000
        duration = 2
        t = torch.linspace(0, duration, sample_rate * duration)
        audio = torch.sin(2 * torch.pi * 440 * t)
        x = audio.unsqueeze(0)  # (1, time)

        with torch.no_grad():
            result = beats_model.extract_embeddings(x)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(emb.shape[0] == 1 for emb in result)
        assert all(torch.is_tensor(emb) for emb in result)
        assert all(not torch.isnan(emb).any() for emb in result)
        assert all(not torch.isinf(emb).any() for emb in result)
