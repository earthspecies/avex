from typing import Optional
from unittest.mock import patch

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

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x


class TestModelBaseHooks:
    """Test the persistent hooks implementation in ModelBase."""

    def test_hook_initialization(self) -> None:
        """Test that hooks are properly initialized."""
        model = TestModel()

        # Check that hook storage is initialized
        assert isinstance(model._hooks, dict)
        assert isinstance(model._hook_outputs, dict)
        assert len(model._hooks) == 0  # No hooks registered initially

        # Trigger linear layer discovery
        model._discover_linear_layers()

        # Check that linear layers are discovered
        assert len(model._linear_layer_names) == 3
        assert "layer1" in model._linear_layer_names
        assert "layer2" in model._linear_layer_names
        assert "output" in model._linear_layer_names

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
        """Test that hooks are registered only once per layer."""
        model = TestModel()

        # Register hooks for specific layers
        model._register_hooks_for_layers(["layer1", "layer2"])

        # Check that hooks are registered
        assert len(model._hooks) == 2
        assert "layer1" in model._hooks
        assert "layer2" in model._hooks

        # Register same layers again - should not add duplicate hooks
        model._register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2  # No additional hooks

    def test_hook_output_capture(self) -> None:
        """Test that hook outputs are captured correctly."""
        model = TestModel()

        # Register hooks
        model._register_hooks_for_layers(["layer1", "layer2"])

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
        """Test that extract_embeddings is more efficient with persistent hooks."""
        model = TestModel()

        # First call - should register hooks
        x = torch.randn(2, 10)
        result1 = model.extract_embeddings(x, ["layer1", "layer2"])
        assert result1.shape[1] == 50  # 20 + 30 dimensions

        # Check that hooks are registered
        assert len(model._hooks) == 2
        assert "layer1" in model._hooks
        assert "layer2" in model._hooks

        # Second call with same layers - should reuse existing hooks
        result2 = model.extract_embeddings(x, ["layer1", "layer2"])
        assert result2.shape[1] == 50

        # Hooks should still be registered (not re-registered)
        assert len(model._hooks) == 2

        # Test with different layers - should add new hooks
        result3 = model.extract_embeddings(x, ["layer1", "layer2", "output"])
        assert result3.shape[1] == 55  # 20 + 30 + 5 dimensions

        # Should now have 3 hooks
        assert len(model._hooks) == 3
        assert "output" in model._hooks

    def test_extract_embeddings_all_layers(self) -> None:
        """Test extract_embeddings with 'all' layers specification."""
        model = TestModel()

        x = torch.randn(2, 10)
        embeddings = model.extract_embeddings(x, ["all"])

        # Should get embeddings from layer1 and layer2 (excluding output)
        assert embeddings.shape[1] == 50  # 20 + 30 dimensions

    def test_hook_cleanup(self) -> None:
        """Test that hooks are properly cleaned up."""
        model = TestModel()

        # Register some hooks
        model._register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2

        # Cleanup hooks
        model._cleanup_hooks()
        assert len(model._hooks) == 0
        assert len(model._hook_outputs) == 0

    def test_clear_hook_outputs(self) -> None:
        """Test that hook outputs are cleared correctly."""
        model = TestModel()

        # Register hooks and capture outputs
        model._register_hooks_for_layers(["layer1"])
        x = torch.randn(2, 10)
        model(x)

        assert len(model._hook_outputs) > 0

        # Clear outputs
        model._clear_hook_outputs()
        assert len(model._hook_outputs) == 0

    def test_invalid_layer_handling(self) -> None:
        """Test handling of invalid layer names."""
        model = TestModel()

        # Try to register hooks for non-existent layer
        with patch("logging.Logger.warning") as mock_warning:
            model._register_hooks_for_layers(["nonexistent_layer"])
            mock_warning.assert_called()

    def test_extract_embeddings_no_layers_found(self) -> None:
        """Test error handling when no layers are found."""
        model = TestModel()

        x = torch.randn(2, 10)

        with pytest.raises(ValueError, match="No layers found matching"):
            model.extract_embeddings(x, ["nonexistent_layer"])
