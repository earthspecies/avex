"""Tests for BEATs model embedding extraction functionality."""

import pytest
import torch

from representation_learning.models.beats_model import Model


class TestBEATsModelEmbeddingExtraction:
    """Test BEATs model embedding extraction functionality."""

    @pytest.fixture(scope="class")
    def beats_model(self) -> Model:
        """Create a BEATs model for testing.

        Model is loaded once per test class to improve performance.

        Returns:
            Model: A configured BEATs model for testing.
        """
        return Model(
            num_classes=10,
            pretrained=True,
            return_features_only=True,
            device="cpu",
            use_naturelm=True,
            disable_layerdrop=True,
        )

    @pytest.fixture(scope="class")
    def beats_model_with_classifier(self) -> Model:
        """Create a BEATs model with classifier for testing.

        Model is loaded once per test class to improve performance.

        Returns:
            Model: A configured BEATs model with classifier for testing.
        """
        return Model(
            num_classes=10,
            pretrained=True,
            return_features_only=False,
            device="cpu",
            use_naturelm=True,
            disable_layerdrop=True,
        )

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
        # Checking request.fixturenames ensures we don't initialize unused fixtures
        if "beats_model" in request.fixturenames:
            beats_model = request.getfixturevalue("beats_model")
            beats_model.deregister_all_hooks()
        if "beats_model_with_classifier" in request.fixturenames:
            beats_model_with_classifier = request.getfixturevalue("beats_model_with_classifier")
            beats_model_with_classifier.deregister_all_hooks()

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor.

        Returns:
            torch.Tensor: Random audio tensor with shape (batch_size, time_steps).
        """
        return torch.randn(2, 16000)  # 2 seconds at 16kHz

    def test_extract_embeddings_basic_functionality(
        self, beats_model: Model, beats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test basic embedding extraction with tensor and dict inputs, both model types."""
        # Test with tensor input
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_tensor = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Test with dict input
        input_dict = {"raw_wav": sample_audio}
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_dict = beats_model.extract_embeddings(input_dict, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Test with classifier model
        beats_model_with_classifier.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_classifier = beats_model_with_classifier.extract_embeddings(sample_audio, aggregation="mean")
        beats_model_with_classifier.deregister_all_hooks()

        # All should work and have correct shapes
        assert embeddings_tensor.shape == (2, 768)
        assert embeddings_dict.shape == (2, 768)
        assert embeddings_classifier.shape[0] == 2
        assert embeddings_classifier.shape[1] > 0
        assert torch.is_tensor(embeddings_tensor)
        assert torch.is_tensor(embeddings_dict)
        assert torch.is_tensor(embeddings_classifier)

    def test_extract_embeddings_layer_selection(self, beats_model: Model, sample_audio: torch.Tensor) -> None:
        """Test extraction from single, multiple, and all layers."""
        # Single layer
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_single = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Multiple layers (if classifier available)
        layers = ["backbone.post_extract_proj"]
        if hasattr(beats_model, "classifier") and beats_model.classifier is not None:
            layers.append("classifier")
        beats_model.register_hooks_for_layers(layers)
        embeddings_multiple = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # All layers
        beats_model._discover_linear_layers()
        all_layers = beats_model._layer_names + ["backbone.post_extract_proj"]
        beats_model.register_hooks_for_layers(all_layers)
        embeddings_all = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # All should work
        assert embeddings_single.shape[0] == 2
        assert embeddings_multiple.shape[0] == 2
        assert embeddings_all.shape[0] == 2
        assert all(emb.shape[1] > 0 for emb in [embeddings_single, embeddings_multiple, embeddings_all])

    def test_extract_embeddings_input_formats_and_padding(self, beats_model: Model, sample_audio: torch.Tensor) -> None:
        """Test extraction with padding mask in both tensor and dict formats."""
        padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
        padding_mask[:, 8000:] = True  # Pad second half

        # Test with padding_mask parameter
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_param = beats_model.extract_embeddings(sample_audio, padding_mask=padding_mask, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Test with padding_mask in dict
        input_dict = {"raw_wav": sample_audio, "padding_mask": padding_mask}
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings_dict = beats_model.extract_embeddings(input_dict, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Both should work
        assert embeddings_param.shape[0] == 2
        assert embeddings_dict.shape[0] == 2
        assert embeddings_param.shape[1] > 0
        assert embeddings_dict.shape[1] > 0

    def test_extract_embeddings_aggregation_modes(self, beats_model: Model, sample_audio: torch.Tensor) -> None:
        """Test different aggregation modes."""
        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        # Mean aggregation
        embeddings_mean = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # None aggregation (returns tensor)
        embeddings_none = beats_model.extract_embeddings(sample_audio, aggregation="none")

        beats_model.deregister_all_hooks()

        # Both should work
        assert torch.is_tensor(embeddings_mean)
        assert torch.is_tensor(embeddings_none)
        assert embeddings_mean.shape[0] == 2
        assert embeddings_none.shape[0] == 2
        assert embeddings_mean.shape[1] > 0
        assert embeddings_none.shape[1] > 0

    def test_extract_embeddings_consistency_and_state_preservation(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test consistency across calls and state preservation."""
        initial_training = beats_model.training

        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        # Multiple calls should be consistent
        embeddings1 = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        embeddings2 = beats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Forward should still work after embedding extraction
        forward_output = beats_model.forward(sample_audio)

        beats_model.deregister_all_hooks()

        # Check consistency
        assert torch.allclose(embeddings1, embeddings2)
        assert embeddings1.shape == embeddings2.shape

        # Check state preservation
        assert beats_model.training == initial_training

        # Forward should work
        assert forward_output.shape[0] == 2
        assert forward_output.shape[1] > 0

    def test_extract_embeddings_gradient_handling(self, beats_model: Model, sample_audio: torch.Tensor) -> None:
        """Test that extraction is gradient-free and handles gradients correctly."""
        sample_audio.requires_grad_(True)

        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        embeddings = beats_model.extract_embeddings(sample_audio, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Should not have gradients
        assert sample_audio.grad is None
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_extract_embeddings_batch_sizes_and_error_handling(
        self, beats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test different batch sizes and error handling."""
        # Different batch sizes
        for batch_size in [1, 2]:
            audio = torch.randn(batch_size, 16000)
            beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
            embeddings = beats_model.extract_embeddings(audio, aggregation="mean")
            beats_model.deregister_all_hooks()
            assert embeddings.shape[0] == batch_size
            assert embeddings.shape[1] > 0

        # Error handling
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(None)

        empty_audio = torch.empty(0)
        with pytest.raises(ValueError):
            beats_model.extract_embeddings(empty_audio)

    def test_extract_embeddings_error_cases(self, beats_model: Model, sample_audio: torch.Tensor) -> None:
        """Test error cases for invalid layers and missing hooks."""
        # Invalid layer
        with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found in model"):
            beats_model.register_hooks_for_layers(["nonexistent_layer"])

        # No hooks registered
        with pytest.raises(ValueError, match="No hooks are registered in the model"):
            beats_model.extract_embeddings(sample_audio, aggregation="mean")

    def test_extract_embeddings_layer_discovery(self, beats_model: Model) -> None:
        """Test that linear layers are properly discovered."""
        beats_model._discover_linear_layers()

        assert hasattr(beats_model, "_layer_names")
        assert isinstance(beats_model._layer_names, list)
        assert len(beats_model._layer_names) > 0
        assert any("backbone" in name for name in beats_model._layer_names)

    def test_deterministic_embeddings_regression(self, beats_model: Model) -> None:
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
        beats_model.eval()

        beats_model.register_hooks_for_layers(["backbone.post_extract_proj"])
        with torch.no_grad():
            embeddings = beats_model.extract_embeddings(signal, aggregation="mean")
        beats_model.deregister_all_hooks()

        # Expected first 20 values (captured with seed=42)
        expected_first_20 = [
            -0.507676899433136,
            -0.1969260573387146,
            0.03937510401010513,
            0.14362706243991852,
            0.13660180568695068,
            -0.04688118025660515,
            -0.08418136835098267,
            0.0059922984801232815,
            -0.021161099895834923,
            -0.10174560546875,
            -0.24325279891490936,
            -0.07431062310934067,
            -0.0019120449433103204,
            0.053144972771406174,
            0.04045940935611725,
            -0.13030825555324554,
            -0.1450606733560562,
            -0.05920398235321045,
            -0.06417909264564514,
            -0.04930436983704567,
        ]

        actual_first_20 = embeddings[0, :20].cpu().numpy().tolist()

        # Use rtol=1e-5, atol=1e-5 for floating point comparison
        np.testing.assert_allclose(
            actual_first_20,
            expected_first_20,
            rtol=1e-5,
            atol=1e-5,
            err_msg="BEATs embeddings do not match expected values",
        )
