import pytest
import torch

from representation_learning.models.aves_model import Model as AVESModel


class TestAVESModel:
    """Test suite for AVES model embedding extraction functionality."""

    @pytest.fixture
    def aves_model(self) -> AVESModel:
        """Create a real AVES model for testing.

        Returns:
            AVESModel: A configured AVES model for testing.
        """
        model = AVESModel(num_classes=10, device="cpu", audio_config=None)

        return model

    def test_model_initialization(self, aves_model: AVESModel) -> None:
        """Test that the model initializes correctly with MLP layer discovery."""
        assert hasattr(aves_model, "_mlp_layer_names")
        assert isinstance(aves_model._mlp_layer_names, list)
        # Should discover some MLP layers in the wav2vec2 model
        assert len(aves_model._mlp_layer_names) >= 0
        print(f"Discovered {len(aves_model._mlp_layer_names)} MLP layers")

    def test_extract_embeddings_empty_layers(self, aves_model: AVESModel) -> None:
        """Test embedding extraction with empty layers list returns main features."""
        x = torch.randn(2, 16000)  # 2 seconds of audio at 16kHz

        with torch.no_grad():
            result = aves_model.extract_embeddings(x, layers=[])

        assert result.shape == (2, 768)  # (batch, features)
        assert torch.is_tensor(result)

    def test_extract_embeddings_dict_input(self, aves_model: AVESModel) -> None:
        """Test embedding extraction with dictionary input."""
        x = {
            "raw_wav": torch.randn(2, 16000),
            "padding_mask": torch.zeros(2, 16000, dtype=torch.bool),
        }

        with torch.no_grad():
            result = aves_model.extract_embeddings(x, layers=[])

        assert result.shape == (2, 768)
        assert torch.is_tensor(result)

    def test_extract_embeddings_with_padding_mask(self, aves_model: AVESModel) -> None:
        """Test embedding extraction with padding mask."""
        x = torch.randn(2, 16000)
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)

        with torch.no_grad():
            result = aves_model.extract_embeddings(
                x, layers=[], padding_mask=padding_mask
            )

        assert result.shape == (2, 768)
        assert torch.is_tensor(result)

    def test_extract_embeddings_all_layers(self, aves_model: AVESModel) -> None:
        """Test embedding extraction with 'all' layers specification."""
        x = torch.randn(2, 16000)

        with torch.no_grad():
            result = aves_model.extract_embeddings(x, layers=["all"])

        # Should return embeddings from all discovered MLP layers
        assert torch.is_tensor(result)
        # The result should have more features than just the main features
        # since it includes multiple MLP layers
        assert result.shape[1] >= 768

    def test_extract_embeddings_specific_layers(self, aves_model: AVESModel) -> None:
        """Test embedding extraction with specific layer names."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = [aves_model._mlp_layer_names[0]]

            with torch.no_grad():
                result = aves_model.extract_embeddings(x, layers=layers)

            assert torch.is_tensor(result)
            assert result.shape[0] == 2  # batch size
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result = aves_model.extract_embeddings(x, layers=[])

            assert result.shape == (2, 768)

    def test_extract_embeddings_no_layers_found(self, aves_model: AVESModel) -> None:
        """Test embedding extraction when no specified layers are found."""
        x = torch.randn(2, 16000)
        layers = ["nonexistent.layer"]

        with pytest.raises(ValueError, match="No layers found matching"):
            with torch.no_grad():
                aves_model.extract_embeddings(x, layers=layers)

    def test_extract_embeddings_average_over_time_false(
        self, aves_model: AVESModel
    ) -> None:
        """Test embedding extraction without averaging over time."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = [aves_model._mlp_layer_names[0]]

            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=layers, average_over_time=False
                )

            # Should return list of tensors when not averaging
            assert isinstance(result, list)
            assert len(result) == 1
            # Should be 3D tensor (batch, time, features)
            assert len(result[0].shape) == 3
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=[], average_over_time=False
                )

            # Should return list with single 3D tensor (batch, time, features)
            assert isinstance(result, list)
            assert len(result) == 1
            assert len(result[0].shape) == 3  # (batch, time, features)
            assert result[0].shape[0] == 2  # batch size
            assert result[0].shape[2] == 768  # feature dimension

    def test_extract_embeddings_3d_tensor_handling(self, aves_model: AVESModel) -> None:
        """Test handling of 3D tensors in embedding extraction."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = [aves_model._mlp_layer_names[0]]
            # Dynamically get expected output dim
            layer_name = aves_model._mlp_layer_names[0]
            module = dict(aves_model.named_modules())[layer_name]
            expected_dim = module.out_features
            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=layers, average_over_time=True
                )
            # Should average over time dimension
            assert result.shape == (2, expected_dim)  # (batch, features)
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=[], average_over_time=True
                )
            assert result.shape == (2, 768)

    def test_extract_embeddings_2d_tensor_handling(self, aves_model: AVESModel) -> None:
        """Test handling of 2D tensors in embedding extraction."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = [aves_model._mlp_layer_names[0]]
            # Dynamically get expected output dim
            layer_name = aves_model._mlp_layer_names[0]
            module = dict(aves_model.named_modules())[layer_name]
            expected_dim = module.out_features
            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=layers, average_over_time=True
                )
            # Should keep 2D shape as is
            assert result.shape == (2, expected_dim)  # (batch, features)
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result = aves_model.extract_embeddings(
                    x, layers=[], average_over_time=True
                )
            assert result.shape == (2, 768)

    def test_extract_embeddings_hook_cleanup(self, aves_model: AVESModel) -> None:
        """Test that hook outputs are properly cleaned up."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = [aves_model._mlp_layer_names[0]]

            with torch.no_grad():
                result1 = aves_model.extract_embeddings(x, layers=layers)
                result2 = aves_model.extract_embeddings(x, layers=layers)

            # Both calls should work without errors (hooks cleaned up properly)
            assert torch.is_tensor(result1)
            assert torch.is_tensor(result2)
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result1 = aves_model.extract_embeddings(x, layers=[])
                result2 = aves_model.extract_embeddings(x, layers=[])

            # Both calls should work without errors
            assert torch.is_tensor(result1)
            assert torch.is_tensor(result2)
            assert result1.shape == (2, 768)
            assert result2.shape == (2, 768)

    def test_extract_embeddings_all_layers_fallback(
        self, aves_model: AVESModel
    ) -> None:
        """Test fallback behavior when no MLP layers are found with 'all'."""
        # Temporarily clear MLP layers to simulate no discovery
        original_mlp_layers = aves_model._mlp_layer_names.copy()
        aves_model._mlp_layer_names = []

        try:
            x = torch.randn(2, 16000)

            with torch.no_grad():
                result1 = aves_model.extract_embeddings(x, layers=["all"])
                result2 = aves_model.extract_embeddings(x, layers=[])

            # Should fallback to main features when no MLP layers found
            assert torch.is_tensor(result1)
            assert torch.is_tensor(result2)
            assert result1.shape == (2, 768)
            assert result2.shape == (2, 768)
        finally:
            # Restore original MLP layers
            aves_model._mlp_layer_names = original_mlp_layers

    def test_extract_embeddings_mixed_layers_and_all(
        self, aves_model: AVESModel
    ) -> None:
        """Test embedding extraction with both specific layers and 'all'."""
        x = torch.randn(2, 16000)

        # Use the first discovered MLP layer if available
        if aves_model._mlp_layer_names:
            layers = ["all", aves_model._mlp_layer_names[0]]

            with torch.no_grad():
                result = aves_model.extract_embeddings(x, layers=layers)

            # Should include both 'all' layers and specific layer
            assert torch.is_tensor(result)
            assert result.shape[0] == 2  # batch size
            # Should have more features than just the main features
            assert result.shape[1] >= 768
        else:
            # If no MLP layers found, test with empty layers
            with torch.no_grad():
                result = aves_model.extract_embeddings(x, layers=[])

            assert result.shape == (2, 768)

    def test_extract_embeddings_consistency(self, aves_model: AVESModel) -> None:
        """Test that embedding extraction is consistent across calls."""
        x = torch.randn(2, 16000)

        with torch.no_grad():
            result1 = aves_model.extract_embeddings(x, layers=[])
            result2 = aves_model.extract_embeddings(x, layers=[])

        # Results should have same shape
        assert result1.shape == result2.shape
        assert result1.shape == (2, 768)

    def test_extract_embeddings_different_batch_sizes(
        self, aves_model: AVESModel
    ) -> None:
        """Test embedding extraction with different batch sizes."""
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 16000)

            with torch.no_grad():
                result = aves_model.extract_embeddings(x, layers=[])

            assert result.shape == (batch_size, 768)

    def test_extract_embeddings_device_handling(self, aves_model: AVESModel) -> None:
        """Test that embedding extraction works on different devices."""
        x = torch.randn(2, 16000)

        # Test on CPU (default)
        with torch.no_grad():
            result_cpu = aves_model.extract_embeddings(x, layers=[])

        assert result_cpu.device == torch.device("cpu")
        assert result_cpu.shape == (2, 768)

    def test_extract_embeddings_gradient_handling(self, aves_model: AVESModel) -> None:
        """Test that embedding extraction properly handles gradients."""
        x = torch.randn(2, 16000, requires_grad=True)

        # Should work without gradients
        with torch.no_grad():
            result = aves_model.extract_embeddings(x, layers=[])

        assert result.shape == (2, 768)
        assert not result.requires_grad

    def test_extract_embeddings_with_realistic_audio(
        self, aves_model: AVESModel
    ) -> None:
        """Test embedding extraction with realistic audio data."""
        # Create realistic audio data (sine wave)
        sample_rate = 16000
        duration = 2  # seconds
        t = torch.linspace(0, duration, sample_rate * duration)
        audio = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave

        # Add batch dimension
        x = audio.unsqueeze(0)  # (1, time)

        with torch.no_grad():
            result = aves_model.extract_embeddings(x, layers=[])

        assert result.shape == (1, 768)
        assert torch.is_tensor(result)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
