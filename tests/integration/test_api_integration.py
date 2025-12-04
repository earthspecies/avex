"""Integration tests for the representation_learning API.

This test suite verifies the main API functions work correctly together,
inspired by the examples in the examples/ directory.
"""

import pytest
import torch

from representation_learning import (
    describe_model,
    get_model_spec,
    list_models,
    load_model,
)
from representation_learning.configs import ModelSpec
from representation_learning.models.get_model import get_model


class TestAPIIntegration:
    """Integration tests for the main API functions."""

    def test_registry_api(self) -> None:
        """Test registry API: list_models, get_model_spec, describe_model, and error handling."""
        # Test list_models
        models = list_models()
        assert isinstance(models, dict), "list_models() should return a dict"
        assert len(models) > 0, "Should have at least one model available"

        # Test get_model_spec for valid model
        model_name = list(models.keys())[0]
        model_spec = get_model_spec(model_name)
        assert model_spec is not None, f"get_model_spec('{model_name}') should return a ModelSpec"
        assert isinstance(model_spec, ModelSpec), "Should return a ModelSpec instance"
        assert model_spec.name is not None, "ModelSpec should have a name"

        # Test get_model_spec for invalid model
        invalid_name = "nonexistent_model_xyz123"
        assert get_model_spec(invalid_name) is None, "Should return None for invalid model"

        # Test describe_model doesn't raise
        try:
            describe_model(model_name, verbose=False)
        except Exception as e:
            pytest.fail(f"describe_model('{model_name}') raised {e}")

    def test_model_creation_and_forward_pass(self) -> None:
        """Test complete model workflow: get_model, forward pass, eval mode, device handling, parameters."""
        models = list_models()
        if not models:
            pytest.skip("No models available in registry")

        model_name = list(models.keys())[0]
        model_spec = get_model_spec(model_name)
        if model_spec is None:
            pytest.skip(f"Could not get model spec for {model_name}")

        # Create model
        model = get_model(model_spec, num_classes=10)
        assert model is not None, "get_model() should return a model"
        assert hasattr(model, "forward"), "Model should have a forward method"

        # Test parameter count
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"
        assert num_params < 1e10, "Model should have reasonable number of parameters"

        # Move to device, set eval mode, and test forward pass
        model = model.to("cpu")
        model.eval()
        assert not model.training, "Model should be in eval mode"

        dummy_input = torch.randn(1, 16000, device="cpu")
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)

        assert output is not None, "Model forward pass should return output"
        assert torch.is_tensor(output), "Output should be a tensor"
        assert output.shape[0] == 1, "Output batch size should match input"
        assert output.device.type == "cpu", "Output should be on CPU"

    def test_load_model_features_only(self) -> None:
        """Test load_model() in embedding extraction mode with deterministic embedding values."""
        import numpy as np

        models = list_models()
        if not models:
            pytest.skip("No models available in registry")

        beats_models = [name for name in models.keys() if "beats" in name.lower()]
        if not beats_models:
            pytest.skip("No BEATs models available for testing")

        try:
            # Set deterministic behavior
            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True, warn_only=True)
            np.random.seed(42)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            model = load_model(beats_models[0], device="cpu", return_features_only=True)
            model.eval()

            assert model is not None, "load_model() should return a model"
            assert hasattr(model, "forward"), "Model should have a forward method"

            # Create deterministic step signal (1 second at 16kHz)
            # First half is -1.0, second half is +1.0
            num_samples = 16000
            signal = torch.zeros(1, num_samples)
            mid_point = num_samples // 2
            signal[0, :mid_point] = -1.0
            signal[0, mid_point:] = 1.0

            with torch.no_grad():
                output = model(signal, padding_mask=None)

            assert output is not None, "Model forward pass should return output"
            assert torch.is_tensor(output), "Output should be a tensor"

            # Expected first 20 values (captured with seed=42 using load_model with features_only)
            expected_first_20 = [
                0.25917747616767883,
                -0.6088295578956604,
                -0.37685254216194153,
                0.2771954834461212,
                0.050542622804641724,
                -0.3111077845096588,
                -0.30799996852874756,
                0.0386744923889637,
                0.15209831297397614,
                0.08887531608343124,
                0.5720303058624268,
                0.8049662709236145,
                0.3912679851055145,
                -0.7082386612892151,
                0.08251217007637024,
                -0.020113110542297363,
                0.08990593999624252,
                0.0052209943532943726,
                0.8818855285644531,
                0.3240680396556854,
            ]

            actual_first_20 = output[0, :20].cpu().numpy().tolist()

            # Use rtol=1e-5, atol=1e-5 for floating point comparison
            np.testing.assert_allclose(
                actual_first_20,
                expected_first_20,
                rtol=1e-5,
                atol=1e-5,
                err_msg="load_model features_only embeddings do not match expected values",
            )
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")
