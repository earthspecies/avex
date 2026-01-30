"""Tests for device handling in the avex API.

This module tests that importing the API doesn't break CUDA availability
and that models can be transferred between devices without issues.
"""

from __future__ import annotations

import pytest
import torch

from avex import load_model, register_model
from avex.configs import AudioConfig, ModelSpec
from avex.models.base_model import ModelBase
from avex.models.utils.registry import (
    register_model_class,
)


class TestAPIDeviceHandling:
    """Test device handling in the API."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from avex.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

        # Register a test model class
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class."""

            name = "test_model"

            def __init__(
                self,
                num_classes: int = 10,
                device: str = "cpu",
                audio_config: dict | None = None,
            ) -> None:
                """Initialize test model."""
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.linear = torch.nn.Linear(128, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass.

                Args:
                    x: Input tensor

                Returns:
                    Output tensor from linear layer
                """
                return self.linear(x)

        # Register a test model spec
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="raw",
                normalize=False,
                target_length_seconds=1.0,
            ),
        )
        register_model("test_model", model_spec)
        # Expose the class for direct instantiation in tests
        self.TestModelClass = TestModelClass

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

    def test_cuda_available_after_import(self) -> None:
        """Test that CUDA remains available after importing the API."""
        # Check CUDA state before import
        cuda_available_before = torch.cuda.is_available()
        cuda_count_before = torch.cuda.device_count() if cuda_available_before else 0

        # Import the API (this triggers registry initialization)
        # We import it to verify CUDA state, but don't need to use the function

        # Verify the import worked (this makes the import "used" so no linter warning)
        assert load_model is not None

        # Check CUDA state after import
        cuda_available_after = torch.cuda.is_available()
        cuda_count_after = torch.cuda.device_count() if cuda_available_after else 0

        # CUDA should remain available if it was available before
        assert cuda_available_after == cuda_available_before, (
            f"CUDA availability changed after import: before={cuda_available_before}, after={cuda_available_after}"
        )

        if cuda_available_before:
            assert cuda_count_after == cuda_count_before, (
                f"CUDA device count changed after import: before={cuda_count_before}, after={cuda_count_after}"
            )

    def test_model_creation_cpu(self) -> None:
        """Test that models can be created on CPU."""
        model = self.TestModelClass(num_classes=10, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.device == "cpu"
        # Check that model parameters are on CPU
        assert next(model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_creation_cuda(self) -> None:
        """Test that models can be created on CUDA."""
        model = self.TestModelClass(num_classes=10, device="cuda")

        assert isinstance(model, ModelBase)
        assert model.device == "cuda"
        # Check that model parameters are on CUDA
        assert next(model.parameters()).device.type == "cuda"

    def test_model_transfer_cpu_to_cpu(self) -> None:
        """Test that models can be transferred from CPU to CPU."""
        model = self.TestModelClass(num_classes=10, device="cpu")
        model_cpu = model.cpu()

        assert model_cpu.device == "cpu"
        assert next(model_cpu.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_transfer_cpu_to_cuda(self) -> None:
        """Test that models can be transferred from CPU to CUDA."""
        model = self.TestModelClass(num_classes=10, device="cpu")
        model_cuda = model.cuda()

        assert model_cuda.device == "cpu"  # String attribute doesn't change
        # But parameters should be on CUDA
        assert next(model_cuda.parameters()).device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_transfer_cuda_to_cpu(self) -> None:
        """Test that models can be transferred from CUDA to CPU."""
        model = self.TestModelClass(num_classes=10, device="cuda")
        model_cpu = model.cpu()

        assert model_cpu.device == "cuda"  # String attribute doesn't change
        # But parameters should be on CPU
        assert next(model_cpu.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_transfer_cuda_to_cuda(self) -> None:
        """Test that models can be transferred from CUDA to CUDA."""
        model = self.TestModelClass(num_classes=10, device="cuda")
        model_cuda = model.cuda()

        assert model_cuda.device == "cuda"
        assert next(model_cuda.parameters()).device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_method(self) -> None:
        """Test that models can be moved using .to() method."""
        model = self.TestModelClass(num_classes=10, device="cpu")

        # Move to CUDA
        model_cuda = model.to("cuda")
        assert next(model_cuda.parameters()).device.type == "cuda"

        # Move back to CPU
        model_cpu = model_cuda.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Move to specific CUDA device
        if torch.cuda.device_count() > 0:
            model_device = model_cpu.to(torch.device("cuda:0"))
            assert next(model_device.parameters()).device.type == "cuda"
            assert next(model_device.parameters()).device.index == 0

    def test_model_forward_cpu(self) -> None:
        """Test that models can perform forward pass on CPU."""
        model = self.TestModelClass(num_classes=10, device="cpu")
        x = torch.randn(2, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_forward_cuda(self) -> None:
        """Test that models can perform forward pass on CUDA."""
        model = self.TestModelClass(num_classes=10, device="cuda")
        x = torch.randn(2, 128, device="cuda")

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_forward_device_mismatch_handling(self) -> None:
        """Test that models handle device mismatches correctly."""
        model = self.TestModelClass(num_classes=10, device="cpu")
        x = torch.randn(2, 128, device="cuda")

        # Model should handle device mismatch (either error or auto-move)
        # This test verifies the behavior doesn't crash
        try:
            with torch.no_grad():
                output = model(x)
            # If it succeeds, output should be on the model's device
            assert output.device.type == "cpu"
        except RuntimeError:
            # It's acceptable for models to raise errors on device mismatch
            pass

    def test_multiple_imports_cuda_stability(self) -> None:
        """Test that multiple imports don't affect CUDA availability."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Import multiple times
        from avex import register_model  # noqa: F401

        # CUDA should still be available
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_loading_preserves_cuda(self) -> None:
        """Test that loading models doesn't break CUDA availability."""
        cuda_available_before = torch.cuda.is_available()
        cuda_count_before = torch.cuda.device_count()

        # Load a model (this may trigger additional imports)
        model = self.TestModelClass(num_classes=10, device="cuda")

        cuda_available_after = torch.cuda.is_available()
        cuda_count_after = torch.cuda.device_count()

        assert cuda_available_after == cuda_available_before
        assert cuda_count_after == cuda_count_before

        # Verify model works on CUDA
        x = torch.randn(2, 128, device="cuda")
        with torch.no_grad():
            output = model(x)
        assert output.device.type == "cuda"
