from typing import Dict

import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel


class TestEfficientNetExtractEmbeddings:
    """Test the extract_embeddings functionality for EfficientNet model."""

    @pytest.fixture(scope="class")
    def model(self) -> EfficientNetModel:
        """Create an EfficientNet model for testing.

        Model is loaded once per test class to improve performance.

        Returns:
            EfficientNetModel: A configured EfficientNet model for testing.
        """
        device = "cpu"
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device=device,
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b0",
        )
        return model

    @pytest.fixture(autouse=True)
    def cleanup_hooks(self, request: pytest.FixtureRequest) -> None:
        """Ensure hooks are cleaned up after each test.

        This fixture runs automatically after each test to ensure hooks
        are deregistered, maintaining test isolation even when using
        class-scoped model fixtures. Only cleans up models that are
        actually requested by the test to avoid unnecessary initialization.

        Args:
            request: Pytest request object to access test fixtures.

        Yields:
            None: Yields control to the test, then cleans up hooks after.
        """
        yield
        # Clean up hooks only for models that are actually requested by the test
        if "model" in request.fixturenames:
            model = request.getfixturevalue("model")
            model.deregister_all_hooks()

    @pytest.fixture
    def audio_input(self) -> torch.Tensor:
        """Create dummy audio input for testing.

        Returns:
            torch.Tensor: Random audio tensor with shape (2, 16000).
        """
        return torch.randn(2, 16000)

    @pytest.fixture
    def dict_input(self, audio_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create dictionary input format for testing.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'raw_wav' and 'padding_mask' keys.
        """
        batch_size = audio_input.shape[0]
        time_steps = audio_input.shape[1]
        padding_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)
        return {"raw_wav": audio_input, "padding_mask": padding_mask}

    def test_extract_embeddings_basic_functionality(
        self, model: EfficientNetModel, audio_input: torch.Tensor, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test basic extraction with single layer, multiple layers, and dict input."""
        model._discover_linear_layers()
        conv_layer_name = model._layer_names[0] if model._layer_names else "model.features.0.0"

        # Single layer
        model.register_hooks_for_layers([conv_layer_name])
        embeddings_single = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Multiple layers
        if len(model._layer_names) >= 2:
            conv_layer_names = model._layer_names[:2]
        else:
            conv_layer_names = [model._layer_names[0], model._layer_names[0]]

        model.register_hooks_for_layers(conv_layer_names)
        embeddings_multiple = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # All layers with dict input
        model.register_hooks_for_layers(model._layer_names)
        embeddings_dict = model.extract_embeddings(x=dict_input, aggregation="mean")
        model.deregister_all_hooks()

        # All should work
        assert embeddings_single.shape[0] == 2
        assert embeddings_multiple.shape[0] == 2
        assert embeddings_dict.shape[0] == 2
        assert all(emb.shape[1] > 0 for emb in [embeddings_single, embeddings_multiple, embeddings_dict])
        assert all(torch.is_tensor(emb) for emb in [embeddings_single, embeddings_multiple, embeddings_dict])

    def test_extract_embeddings_all_layers_and_layer_discovery(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test extraction with all layers and verify layer discovery."""
        model._discover_linear_layers()
        assert hasattr(model, "_layer_names")
        assert len(model._layer_names) > 0

        # All layers
        model.register_hooks_for_layers(model._layer_names)
        embeddings_all = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Subset comparison
        subset_layers = model._layer_names[:3] if len(model._layer_names) >= 3 else model._layer_names
        model.register_hooks_for_layers(subset_layers)
        embeddings_subset = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings_all.shape[0] == 2
        assert embeddings_all.shape[1] > 1000
        assert embeddings_subset.shape[1] < embeddings_all.shape[1]

    def test_extract_embeddings_aggregation_modes(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test different aggregation modes."""
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        # Mean aggregation
        embeddings_mean = model.extract_embeddings(x=audio_input, aggregation="mean")

        # Max aggregation
        embeddings_max = model.extract_embeddings(x=audio_input, aggregation="max")

        # CLS token aggregation
        embeddings_cls = model.extract_embeddings(x=audio_input, aggregation="cls_token")

        # None aggregation
        conv_layer_name = model._layer_names[0] if model._layer_names else "model.features.0.0"
        model.deregister_all_hooks()
        model.register_hooks_for_layers([conv_layer_name])
        embeddings_none = model.extract_embeddings(x=audio_input, aggregation="none")
        model.deregister_all_hooks()

        # All should work
        assert all(torch.is_tensor(emb) for emb in [embeddings_mean, embeddings_max, embeddings_cls, embeddings_none])
        assert all(emb.shape[0] == 2 for emb in [embeddings_mean, embeddings_max, embeddings_cls, embeddings_none])
        assert embeddings_mean.shape[1] > 1000
        assert embeddings_max.shape[1] > 1000
        assert embeddings_cls.shape[1] > 1000
        assert embeddings_none.shape[1] > 0

    def test_extract_embeddings_consistency(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings produces consistent results."""
        torch.manual_seed(42)

        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)

        embeddings1 = model.extract_embeddings(x=audio_input, aggregation="mean")
        embeddings2 = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.shape[0] == 2
        assert embeddings1.shape[1] > 1000
        assert not torch.isnan(embeddings1).any()
        assert not torch.isnan(embeddings2).any()
        assert not torch.isinf(embeddings1).any()
        assert not torch.isinf(embeddings2).any()
        assert embeddings1.abs().max() < 1000
        assert embeddings2.abs().max() < 1000

    def test_extract_embeddings_padding_mask(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test padding mask handling."""
        batch_size = audio_input.shape[0]
        time_steps = audio_input.shape[1]
        padding_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
        padding_mask[:, -100:] = False

        dict_input = {"raw_wav": audio_input, "padding_mask": padding_mask}

        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)
        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 1000
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_error_handling(self, model: EfficientNetModel, audio_input: torch.Tensor) -> None:
        """Test error handling for invalid layers and aggregation methods."""
        # Invalid layer
        with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found in model"):
            model.register_hooks_for_layers(["nonexistent_layer"])

        # Invalid aggregation
        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)
        with pytest.raises(ValueError, match="Unsupported aggregation method: invalid"):
            model.extract_embeddings(x=audio_input, aggregation="invalid")
        model.deregister_all_hooks()

    def test_extract_embeddings_features_only_mode(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings when model is in features_only mode."""
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

        features_model._discover_linear_layers()
        features_model.register_hooks_for_layers(features_model._layer_names)
        embeddings = features_model.extract_embeddings(x=audio_input, aggregation="mean")
        features_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_gradient_checkpointing(
        self, model: EfficientNetModel, audio_input: torch.Tensor
    ) -> None:
        """Test that gradient checkpointing works with extract_embeddings."""
        model.enable_gradient_checkpointing()
        model.train()

        model._discover_linear_layers()
        model.register_hooks_for_layers(model._layer_names)
        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
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

        model_b1._discover_linear_layers()
        model_b1.register_hooks_for_layers(model_b1._layer_names)
        embeddings_b1 = model_b1.extract_embeddings(x=audio_input, aggregation="mean")
        model_b1.deregister_all_hooks()

        assert embeddings_b1.shape[0] == 2
        assert embeddings_b1.shape[1] > 0
        assert torch.is_tensor(embeddings_b1)

    def test_extract_embeddings_device_consistency(self, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
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

        torch.manual_seed(42)
        model_gpu = EfficientNetModel(
            num_classes=1000,
            pretrained=False,
            device="cuda",
            audio_config=audio_config,
            return_features_only=False,
            efficientnet_variant="b0",
        )

        audio_input_gpu = audio_input.cuda()

        model_cpu._discover_linear_layers()
        model_cpu.register_hooks_for_layers(model_cpu._layer_names)
        embeddings1 = model_cpu.extract_embeddings(x=audio_input, aggregation="mean")
        model_cpu.deregister_all_hooks()

        model_gpu._discover_linear_layers()
        model_gpu.register_hooks_for_layers(model_gpu._layer_names)
        embeddings2 = model_gpu.extract_embeddings(x=audio_input_gpu, aggregation="mean")
        model_gpu.deregister_all_hooks()

        assert embeddings2.device.type == "cuda"
        embeddings2 = embeddings2.cpu()

        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.shape[0] == 2
        assert embeddings1.shape[1] > 1000
        assert not torch.isnan(embeddings1).any()
        assert not torch.isnan(embeddings2).any()
        assert not torch.isinf(embeddings1).any()
        assert not torch.isinf(embeddings2).any()
        assert embeddings1.abs().max() < 1000
        assert embeddings2.abs().max() < 1000

    def test_deterministic_embeddings_regression(self, model: EfficientNetModel) -> None:
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

        model._discover_linear_layers()
        conv_layer_name = model._layer_names[0] if model._layer_names else "model.features.0.0"
        model.register_hooks_for_layers([conv_layer_name])
        with torch.no_grad():
            embeddings = model.extract_embeddings(signal, aggregation="mean")
        model.deregister_all_hooks()

        # Expected first 20 values (captured with seed=42)
        expected_first_20 = [
            -0.8009154796600342,
            -0.30336713790893555,
            -0.05972571671009064,
            -0.0574716180562973,
            -0.055956680327653885,
            -0.05480717122554779,
            -0.053860507905483246,
            -0.052935197949409485,
            -0.05224410817027092,
            -0.0515759214758873,
            -0.0510500930249691,
            -0.05048634484410286,
            -0.0499289445579052,
            -0.04944973438978195,
            -0.04905107617378235,
            -0.04872405529022217,
            -0.048297543078660965,
            -0.04796253517270088,
            -0.047591447830200195,
            -0.04731076583266258,
        ]

        actual_first_20 = embeddings[0, :20].cpu().numpy().tolist()

        # Use rtol=1e-5, atol=1e-5 for floating point comparison
        np.testing.assert_allclose(
            actual_first_20,
            expected_first_20,
            rtol=1e-5,
            atol=1e-5,
            err_msg="EfficientNet embeddings do not match expected values",
        )
