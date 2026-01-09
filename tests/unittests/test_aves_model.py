import pytest
import torch

from avex.models.aves_model import Model as AVESModel
from tests.utils.test_utils import create_cleanup_hooks_fixture


class TestAVESModel:
    """Test suite for AVES model embedding extraction functionality."""

    @pytest.fixture(scope="class")
    def aves_model(self) -> AVESModel:
        """Create a real AVES model for testing.

        Model is loaded once per test class to improve performance.

        Returns:
            AVESModel: A configured AVES model for testing.
        """
        model = AVESModel(num_classes=10, device="cpu", audio_config=None)
        return model

    # Cleanup hooks after each test using shared utility
    cleanup_hooks = create_cleanup_hooks_fixture(model_fixture_name="aves_model")

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor.

        Returns:
            torch.Tensor: Random audio tensor with shape (2, 16000).
        """
        return torch.randn(2, 16000)

    def test_model_initialization_and_layer_discovery(self, aves_model: AVESModel) -> None:
        """Test model initialization and layer discovery."""
        aves_model._discover_linear_layers()
        assert hasattr(aves_model, "_layer_names")
        assert isinstance(aves_model._layer_names, list)
        assert len(aves_model._layer_names) >= 0

    def test_extract_embeddings_input_formats(self, aves_model: AVESModel, sample_audio: torch.Tensor) -> None:
        """Test extraction with tensor, dict, and padding mask inputs."""
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        # Tensor input
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_tensor = aves_model.extract_embeddings(sample_audio, aggregation="mean")
        aves_model.deregister_all_hooks()

        # Dict input
        dict_input = {
            "raw_wav": torch.randn(2, 16000),
            "padding_mask": torch.zeros(2, 16000, dtype=torch.bool),
        }
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_dict = aves_model.extract_embeddings(dict_input, aggregation="mean")
        aves_model.deregister_all_hooks()

        # Padding mask parameter
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_padding = aves_model.extract_embeddings(sample_audio, padding_mask=padding_mask, aggregation="mean")
        aves_model.deregister_all_hooks()

        # All should work
        for result in [result_tensor, result_dict, result_padding]:
            assert result.shape[0] == 2
            assert result.shape[1] > 0
            assert torch.is_tensor(result)

    def test_extract_embeddings_layer_selection(self, aves_model: AVESModel, sample_audio: torch.Tensor) -> None:
        """Test extraction with specific layers and all layers."""
        aves_model._discover_linear_layers()
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        # Specific layer
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_specific = aves_model.extract_embeddings(sample_audio, aggregation="mean")
        aves_model.deregister_all_hooks()

        # All layers (if discovered)
        if aves_model._layer_names:
            aves_model.register_hooks_for_layers(aves_model._layer_names)
            with torch.no_grad():
                result_all = aves_model.extract_embeddings(sample_audio, aggregation="mean")
            aves_model.deregister_all_hooks()

            assert torch.is_tensor(result_all)
            assert result_all.shape[1] >= 768

        assert result_specific.shape[0] == 2
        assert result_specific.shape[1] > 0

    def test_extract_embeddings_aggregation_modes(self, aves_model: AVESModel, sample_audio: torch.Tensor) -> None:
        """Test mean and none aggregation modes."""
        aves_model._discover_linear_layers()
        layer_name = (
            aves_model._layer_names[0]
            if aves_model._layer_names
            else "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"
        )

        # Mean aggregation
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_mean = aves_model.extract_embeddings(sample_audio, aggregation="mean")
        aves_model.deregister_all_hooks()

        # None aggregation
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_none = aves_model.extract_embeddings(sample_audio, aggregation="none")
        aves_model.deregister_all_hooks()

        assert torch.is_tensor(result_mean)
        assert torch.is_tensor(result_none)
        assert result_mean.shape[0] == 2
        assert result_none.dim() == 3  # (batch, time, features)

    def test_extract_embeddings_consistency_and_hook_cleanup(
        self, aves_model: AVESModel, sample_audio: torch.Tensor
    ) -> None:
        """Test consistency across calls and hook cleanup."""
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result1 = aves_model.extract_embeddings(sample_audio, aggregation="mean")
            result2 = aves_model.extract_embeddings(sample_audio, aggregation="mean")
        aves_model.deregister_all_hooks()

        # Results should be consistent
        assert result1.shape == result2.shape
        assert result1.shape[0] == 2
        assert result1.shape[1] > 0

    def test_extract_embeddings_batch_sizes(self, aves_model: AVESModel) -> None:
        """Test extraction with different batch sizes."""
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 16000)
            aves_model.register_hooks_for_layers([layer_name])
            with torch.no_grad():
                result = aves_model.extract_embeddings(x, aggregation="mean")
            aves_model.deregister_all_hooks()

            assert result.shape[0] == batch_size
            assert result.shape[1] > 0

    def test_extract_embeddings_error_handling(self, aves_model: AVESModel) -> None:
        """Test error handling for invalid layers."""
        with pytest.raises(ValueError, match="Layer 'nonexistent.layer' not found in model"):
            aves_model.register_hooks_for_layers(["nonexistent.layer"])

    def test_extract_embeddings_gradient_and_device_handling(
        self, aves_model: AVESModel, sample_audio: torch.Tensor
    ) -> None:
        """Test gradient handling and device consistency."""
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        # Gradient handling
        x_grad = torch.randn(2, 16000, requires_grad=True)
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_grad = aves_model.extract_embeddings(x_grad, aggregation="mean")
        aves_model.deregister_all_hooks()

        assert not result_grad.requires_grad
        assert result_grad.shape[0] == 2

        # Device handling
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result_cpu = aves_model.extract_embeddings(sample_audio, aggregation="mean")
        aves_model.deregister_all_hooks()

        assert result_cpu.device == torch.device("cpu")
        assert result_cpu.shape[0] == 2

    def test_extract_embeddings_realistic_audio(self, aves_model: AVESModel) -> None:
        """Test extraction with realistic audio data."""
        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"

        # Create realistic audio (sine wave)
        sample_rate = 16000
        duration = 2
        t = torch.linspace(0, duration, sample_rate * duration)
        audio = torch.sin(2 * torch.pi * 440 * t)
        x = audio.unsqueeze(0)  # (1, time)

        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            result = aves_model.extract_embeddings(x, aggregation="mean")
        aves_model.deregister_all_hooks()

        assert result.shape[0] == 1
        assert result.shape[1] > 0
        assert torch.is_tensor(result)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_deterministic_embeddings_regression(self, aves_model: AVESModel) -> None:
        """Regression test: verify embeddings match expected values for deterministic step signal.

        This test uses a fixed step signal and deterministic settings to ensure
        model outputs remain consistent across runs.
        """
        import numpy as np

        # Set deterministic behavior
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)
        np.random.seed(42)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Create deterministic step signal (1 second at 16kHz)
        # First half is -1.0, second half is +1.0
        num_samples = 16000
        signal = torch.zeros(1, num_samples)
        mid_point = num_samples // 2
        signal[0, :mid_point] = -1.0
        signal[0, mid_point:] = 1.0

        # Ensure model is in eval mode
        aves_model.eval()

        layer_name = "model.encoder.transformer.layers.0.feed_forward.intermediate_dense"
        aves_model.register_hooks_for_layers([layer_name])
        with torch.no_grad():
            embeddings = aves_model.extract_embeddings(signal, aggregation="mean")
        aves_model.deregister_all_hooks()

        # Expected first 20 values (captured with seed=42)
        expected_first_20 = [
            -0.35211431980133057,
            -1.9091869592666626,
            -0.36330243945121765,
            0.5688936710357666,
            -3.355074405670166,
            -1.5228770971298218,
            -2.6351702213287354,
            -2.245500326156616,
            -1.618971586227417,
            -2.0088589191436768,
            -0.693280041217804,
            -3.16288685798645,
            -1.8873447179794312,
            -0.6843950748443604,
            -3.0951995849609375,
            -1.0851454734802246,
            -0.7989563345909119,
            0.426406592130661,
            -1.6272715330123901,
            -4.18255090713501,
        ]

        actual_first_20 = embeddings[0, :20].cpu().numpy().tolist()

        # Use rtol=1e-5, atol=1e-5 for floating point comparison
        np.testing.assert_allclose(
            actual_first_20,
            expected_first_20,
            rtol=1e-5,
            atol=1e-5,
            err_msg="AVES embeddings do not match expected values",
        )
