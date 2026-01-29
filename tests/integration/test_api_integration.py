"""Integration tests for the avex API.

This test suite verifies the main API functions work correctly together,
inspired by the examples in the examples/ directory.
"""

import pytest
import torch

from avex import (
    describe_model,
    get_model_spec,
    list_models,
    load_model,
)
from avex.configs import ModelSpec, ProbeConfig
from avex.models.probes.utils import build_probe_from_config
from avex.models.utils.factory import build_model_from_spec


class TestAPIIntegration:
    """Integration tests for the main API functions."""

    @pytest.fixture(scope="class")
    def beats_model_name(self) -> str:
        """Shared fixture to get a BEATs model name for resource sharing across tests.

        Returns
        -------
        str
            The name of a BEATs model from the registry.
        """
        models = list_models()
        assert len(models) > 0, "Registry should contain at least one model"
        beats_models = [name for name in models.keys() if "beats" in name.lower()]
        assert len(beats_models) > 0, "Registry should contain at least one BEATs model"
        return beats_models[0]

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

    def test_model_creation_and_forward_pass(self, beats_model_name: str) -> None:
        """Test complete model workflow: model factory, forward pass, eval mode, device handling, parameters.

        Uses BEATs model to share resources with other tests in this class.
        """
        model_spec = get_model_spec(beats_model_name)
        assert model_spec is not None, f"Model spec should exist for '{beats_model_name}'"

        # Create backbone model
        model = build_model_from_spec(model_spec, device="cpu")
        assert model is not None, "model factory should return a model"
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
        """Test load_model() in embedding extraction mode (basic sanity checks)."""
        models = list_models()
        assert len(models) > 0, "Registry should contain at least one model"

        beats_models = [name for name in models.keys() if "beats" in name.lower()]
        assert len(beats_models) > 0, "Registry should contain at least one BEATs model"

        model = load_model(beats_models[0], device="cpu", return_features_only=True)
        model.eval()

        assert model is not None, "load_model() should return a model"
        assert hasattr(model, "forward"), "Model should have a forward method"

        # Create step signal (1 second at 16kHz)
        num_samples = 16000
        signal = torch.zeros(1, num_samples)
        mid_point = num_samples // 2
        signal[0, :mid_point] = -1.0
        signal[0, mid_point:] = 1.0

        with torch.no_grad():
            output = model(signal, padding_mask=None)

        assert output is not None, "Model forward pass should return output"
        assert torch.is_tensor(output), "Output should be a tensor"
        assert output.shape[0] == 1, "Batch dimension should be 1"
        assert output.ndim == 3, "Embedding output should be (batch, time, features)"

    def test_backbone_with_linear_probe(self, beats_model_name: str) -> None:
        """Test attaching a simple linear probe head to a backbone.

        Uses BEATs model to share resources with other tests in this class.
        """
        model_spec = get_model_spec(beats_model_name)
        assert model_spec is not None, f"Model spec should exist for '{beats_model_name}'"

        # Build BEATs backbone
        backbone = build_model_from_spec(model_spec, device="cpu")
        backbone.eval()

        # Attach a linear probe on top of the last layer
        probe_cfg = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )
        probe = build_probe_from_config(
            probe_config=probe_cfg,
            base_model=backbone,
            num_classes=3,
            device="cpu",
        )
        probe.eval()

        dummy_input = torch.randn(2, 16000 * 5)
        with torch.no_grad():
            logits = probe(dummy_input)

        assert logits is not None
        assert torch.is_tensor(logits)
        assert logits.shape == (2, 3)
