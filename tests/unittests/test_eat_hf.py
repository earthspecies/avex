from typing import Dict

import pytest
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.eat_hf import Model as EATHFModel


class TestEATHFExtractEmbeddings:
    """Test the extract_embeddings functionality for EAT HF model."""

    @pytest.fixture(scope="class")
    def model(self) -> EATHFModel:
        """Create an EAT HF model for testing.

        Model is loaded once per test class to improve performance.

        Returns:
            EATHFModel: A configured EAT HF model for testing.
        """
        device = "cpu"
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
        self, model: EATHFModel, audio_input: torch.Tensor, dict_input: Dict[str, torch.Tensor]
    ) -> None:
        """Test basic extraction with single layer, multiple layers, and dict input."""
        # Single layer (fc1)
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc1"])
        embeddings_fc1 = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Single layer (fc2)
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc2"])
        embeddings_fc2 = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Multiple layers
        model.register_hooks_for_layers(
            [
                "backbone.model.blocks.0.mlp.fc1",
                "backbone.model.blocks.0.mlp.fc2",
            ]
        )
        embeddings_multiple = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Dict input with all layers
        model.register_hooks_for_layers(["all"])
        embeddings_dict = model.extract_embeddings(x=dict_input, aggregation="mean")
        model.deregister_all_hooks()

        # All should work
        assert embeddings_fc1.shape == (2, 3072)
        assert embeddings_fc2.shape == (2, 768)
        assert embeddings_multiple.shape == (2, 3840)  # 3072 + 768
        assert embeddings_dict.shape[0] == 2
        assert embeddings_dict.shape[1] > 0
        assert all(
            torch.is_tensor(emb) for emb in [embeddings_fc1, embeddings_fc2, embeddings_multiple, embeddings_dict]
        )

    def test_extract_embeddings_all_layers_and_classifier(self, model: EATHFModel, audio_input: torch.Tensor) -> None:
        """Test extraction with all layers and classifier."""
        # All layers
        model.register_hooks_for_layers(["all"])
        embeddings_all = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Classifier only
        model.register_hooks_for_layers(["classifier"])
        embeddings_classifier = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # Mixed MLP and classifier
        model.register_hooks_for_layers(
            [
                "backbone.model.blocks.0.mlp.fc1",
                "classifier",
            ]
        )
        embeddings_mixed = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        # All should work
        assert embeddings_all.shape[0] == 2
        assert embeddings_classifier.shape == (2, 10)
        assert embeddings_mixed.shape == (2, 3082)  # 3072 + 10
        assert all(torch.is_tensor(emb) for emb in [embeddings_all, embeddings_classifier, embeddings_mixed])

    def test_extract_embeddings_aggregation_modes(self, model: EATHFModel, audio_input: torch.Tensor) -> None:
        """Test different aggregation modes."""
        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc2"])

        # Mean aggregation
        embeddings_mean = model.extract_embeddings(x=audio_input, aggregation="mean")

        # None aggregation (3D tensor)
        embeddings_none = model.extract_embeddings(x=audio_input, aggregation="none")

        model.deregister_all_hooks()

        assert torch.is_tensor(embeddings_mean)
        assert torch.is_tensor(embeddings_none)
        assert embeddings_mean.shape == (2, 768)
        assert embeddings_none.shape == (2, 513, 768)  # (batch, seq_len, features)

    def test_extract_embeddings_consistency(self, model: EATHFModel, audio_input: torch.Tensor) -> None:
        """Test that extract_embeddings produces consistent results."""
        torch.manual_seed(42)

        model.register_hooks_for_layers(["all"])
        embeddings1 = model.extract_embeddings(x=audio_input, aggregation="mean")
        embeddings2 = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        assert torch.allclose(embeddings1, embeddings2, atol=1e-6)

    def test_extract_embeddings_padding_mask(self, model: EATHFModel, audio_input: torch.Tensor) -> None:
        """Test padding mask handling."""
        batch_size = audio_input.shape[0]
        time_steps = audio_input.shape[1]
        padding_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
        padding_mask[:, -100:] = False

        dict_input = {"raw_wav": audio_input, "padding_mask": padding_mask}

        model.register_hooks_for_layers(["all"])
        embeddings = model.extract_embeddings(x=dict_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_error_handling(self, model: EATHFModel, audio_input: torch.Tensor) -> None:
        """Test error handling for invalid layers."""
        with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found in model"):
            model.register_hooks_for_layers(["nonexistent_layer"])

    def test_extract_embeddings_layer_discovery_and_last_layer(
        self, model: EATHFModel, audio_input: torch.Tensor
    ) -> None:
        """Test layer discovery and last_layer special case."""
        # Layer discovery
        model._discover_linear_layers()
        assert hasattr(model, "_layer_names")
        assert len(model._layer_names) > 0

        # Test with discovered layers
        model.register_hooks_for_layers(model._layer_names)
        embeddings = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()
        assert torch.is_tensor(embeddings)

        # Last layer
        model.register_hooks_for_layers(["last_layer"])
        embeddings_last = model.extract_embeddings(x=audio_input, aggregation="mean")
        model.deregister_all_hooks()

        assert embeddings_last.shape[0] == 2
        assert embeddings_last.shape[1] > 0
        assert len(model._hook_layers) == 1
        assert "last_layer" not in model._hook_layers

    def test_extract_embeddings_features_only_mode(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings when model is in features_only mode."""
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

        features_model.register_hooks_for_layers(["all"])
        embeddings = features_model.extract_embeddings(x=audio_input, aggregation="mean")
        features_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        assert torch.is_tensor(embeddings)

    def test_extract_embeddings_custom_num_classes(self, audio_input: torch.Tensor) -> None:
        """Test extract_embeddings with custom number of classes."""
        audio_config = AudioConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
        )

        model_custom = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=5,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            target_length=1024,
            pooling="cls",
        )

        model_custom.register_hooks_for_layers(["classifier"])
        embeddings = model_custom.extract_embeddings(x=audio_input, aggregation="mean")
        model_custom.deregister_all_hooks()

        assert embeddings.shape == (2, 5)
        assert torch.is_tensor(embeddings)

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

        model_cpu = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=10,
            device="cpu",
            audio_config=audio_config,
            return_features_only=False,
            target_length=1024,
            pooling="cls",
        )

        torch.manual_seed(42)
        model_gpu = EATHFModel(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            num_classes=10,
            device="cuda",
            audio_config=audio_config,
            return_features_only=False,
            target_length=1024,
            pooling="cls",
        )

        audio_input_gpu = audio_input.cuda()

        model_cpu.register_hooks_for_layers(["all"])
        embeddings1 = model_cpu.extract_embeddings(x=audio_input, aggregation="mean")
        model_cpu.deregister_all_hooks()

        model_gpu.register_hooks_for_layers(["all"])
        embeddings2 = model_gpu.extract_embeddings(x=audio_input_gpu, aggregation="mean")
        model_gpu.deregister_all_hooks()

        assert embeddings2.device.type == "cuda"
        embeddings2 = embeddings2.cpu()

        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.shape[0] == 2
        assert embeddings1.shape[1] > 0
        assert not torch.isnan(embeddings1).any()
        assert not torch.isnan(embeddings2).any()
        assert not torch.isinf(embeddings1).any()
        assert not torch.isinf(embeddings2).any()
        assert embeddings1.abs().max() < 1000
        assert embeddings2.abs().max() < 1000

    def test_deterministic_embeddings_regression(self, model: EATHFModel) -> None:
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
        model.eval()

        model.register_hooks_for_layers(["backbone.model.blocks.0.mlp.fc1"])
        with torch.no_grad():
            embeddings = model.extract_embeddings(signal, aggregation="mean")
        model.deregister_all_hooks()

        # Expected first 20 values (captured with seed=42)
        expected_first_20 = [
            -0.280917227268219,
            -1.1888835430145264,
            0.0036318127531558275,
            -0.6345792412757874,
            -0.9963047504425049,
            -0.4221092760562897,
            0.007080540992319584,
            -0.1600695699453354,
            -0.8976061344146729,
            -0.0987534299492836,
            -1.3774746656417847,
            -0.002663534367457032,
            -0.9980361461639404,
            0.07578037679195404,
            -0.5971910953521729,
            -0.5333408117294312,
            0.08951346576213837,
            -1.3790560960769653,
            -0.6064006090164185,
            -1.471298098564148,
        ]

        actual_first_20 = embeddings[0, :20].cpu().numpy().tolist()

        # Use rtol=1e-5, atol=1e-5 for floating point comparison
        np.testing.assert_allclose(
            actual_first_20,
            expected_first_20,
            rtol=1e-5,
            atol=1e-5,
            err_msg="EAT-HF embeddings do not match expected values",
        )
