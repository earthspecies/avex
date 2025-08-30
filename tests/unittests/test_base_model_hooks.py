from typing import Optional

import pytest
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase


class TestModel(ModelBase):
    """Simple test model for testing hook functionality."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.output = nn.Linear(30, 5)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # Discover layers for hook management
        self._hook_layers = ["layer1", "layer2", "output"]

        # Discover linear layers for 'all' functionality
        self._discover_linear_layers()

        # Register hooks for the discovered layers
        self.register_hooks_for_layers(self._hook_layers)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

    def _get_linear_layers_excluding_last(self) -> list[str]:
        """Get linear layers excluding the last one.

        Returns:
            list[str]: List of linear layer names excluding the last one.
        """
        linear_layers = self._get_all_linear_layers()
        return linear_layers[:-1] if linear_layers else []


class TestModelBaseHooks:
    """Test the persistent hooks implementation in ModelBase."""

    def test_hook_initialization(self) -> None:
        """Test that hooks are properly initialized."""
        model = TestModel()

        # Check that hook storage is initialized
        assert isinstance(model._hooks, dict)
        assert isinstance(model._hook_outputs, dict)
        # Hooks are now registered in __init__
        assert len(model._hooks) == 3  # Hooks registered for layer1, layer2, output
        assert len(model._hook_layers) == 3  # Layers discovered

        # Check that linear layers are discovered
        assert len(model._layer_names) == 3
        assert "layer1" in model._layer_names
        assert "layer2" in model._layer_names
        assert "output" in model._layer_names

    def test_linear_layer_discovery(self) -> None:
        """Test that linear layers are correctly discovered and cached."""
        model = TestModel()

        # Test getting linear layers excluding last
        linear_layers = model._get_linear_layers_excluding_last()
        assert len(linear_layers) == 2
        assert "layer1" in linear_layers
        assert "layer2" in linear_layers
        assert "output" not in linear_layers  # Last layer excluded

    def test_hook_registration(self) -> None:
        """Test that hooks can be registered on-demand."""
        model = TestModel()

        # Initially hooks are registered from __init__
        assert len(model._hooks) == 3

        # Register hooks for different layers - should clear and re-register
        model.register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2
        assert "layer1" in model._hooks
        assert "layer2" in model._hooks

        # Register same layers again - should clear and re-register
        model.register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2  # Same count

    def test_hook_output_capture(self) -> None:
        """Test that hook outputs are captured correctly."""
        model = TestModel()

        # Hooks are already registered from __init__
        assert "layer1" in model._hooks
        assert "layer2" in model._hooks

        # Create test input
        x = torch.randn(2, 10)

        # Forward pass
        model(x)

        # Check that outputs are captured
        assert "layer1" in model._hook_outputs
        assert "layer2" in model._hook_outputs
        assert model._hook_outputs["layer1"].shape == (2, 20)
        assert model._hook_outputs["layer2"].shape == (2, 30)

    def test_extract_embeddings_efficiency(self) -> None:
        """Test that extract_embeddings works with pre-registered hooks."""
        model = TestModel()

        # Hooks are already registered from __init__
        assert len(model._hooks) == 3

        # First call - should work with existing hooks
        x = torch.randn(2, 10)
        result1 = model.extract_embeddings(x, aggregation="mean")

        # Second call - should work without re-registering hooks
        result2 = model.extract_embeddings(x, aggregation="mean")

        # Results should be the same
        assert torch.allclose(result1, result2)

        # Hooks should still be registered
        assert len(model._hooks) == 3

    def test_extract_embeddings_all_layers(self) -> None:
        """Test extract_embeddings with 'all' layers specification."""
        model = TestModel()

        x = torch.randn(2, 10)
        embeddings = model.extract_embeddings(x, aggregation="mean")

        # Should extract from all registered hooks (layer1, layer2, output)
        assert embeddings.shape[1] == 55  # 20 + 30 + 5 dimensions

    def test_hook_cleanup(self) -> None:
        """Test that hooks are properly cleaned up."""
        model = TestModel()

        # Check that hooks are registered from __init__
        assert len(model._hooks) == 3

        # Cleanup hooks
        model.deregister_all_hooks()
        assert len(model._hooks) == 0
        assert len(model._hook_outputs) == 0

    def test_clear_hook_outputs(self) -> None:
        """Test that hook outputs are cleared correctly."""
        model = TestModel()

        # Hooks are already registered from __init__
        assert "layer1" in model._hooks

        # Capture outputs
        x = torch.randn(2, 10)
        model(x)

        assert len(model._hook_outputs) > 0

        # Clear outputs
        model._clear_hook_outputs()
        assert len(model._hook_outputs) == 0

    def test_extract_embeddings_no_layers_found(self) -> None:
        """Test error handling when no hooks are registered."""
        model = TestModel()

        # Remove all hooks
        model.deregister_all_hooks()
        assert len(model._hooks) == 0

        x = torch.randn(2, 10)

        with pytest.raises(ValueError, match="No hooks registered"):
            model.extract_embeddings(x)

    def test_hook_persistence(self) -> None:
        """Test that hooks persist across multiple forward passes."""
        model = TestModel()

        # Hooks are registered in __init__
        initial_hook_count = len(model._hooks)
        assert initial_hook_count == 3

        # Multiple forward passes
        x = torch.randn(2, 10)
        for _ in range(3):
            model(x)
            # Hooks should remain registered
            assert len(model._hooks) == initial_hook_count

    def test_hook_output_clearing(self) -> None:
        """Test that hook outputs are cleared between calls but hooks remain."""
        model = TestModel()

        x = torch.randn(2, 10)

        # First call
        result1 = model.extract_embeddings(x, aggregation="mean")

        # Second call
        result2 = model.extract_embeddings(x, aggregation="mean")

        # Results should be the same
        assert torch.allclose(result1, result2)

        # Hooks should still be registered
        assert len(model._hooks) == 3
