"""Tests for model pruning functionality."""

import pytest
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.get_probe import (
    prune_model_to_layer,
)


class SimpleTestModel(ModelBase):
    """A simple test model with multiple layers for testing pruning."""

    def __init__(self, device: str = "cpu", audio_config: dict | None = None) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Create a simple model with multiple layers
        self.layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Dropout(0.1))
        self.layer2 = nn.Sequential(nn.Linear(20, 15), nn.ReLU(), nn.Dropout(0.1))
        self.layer3 = nn.Sequential(nn.Linear(15, 8), nn.ReLU(), nn.Dropout(0.1))
        self.final_layer = nn.Linear(8, 5)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Simple forward pass through all layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_layer(x)
        return x


class TestModelPruning:
    """Test cases for model pruning functionality."""

    def test_prune_model_to_layer_basic(self) -> None:
        """Test basic model pruning functionality."""
        model = SimpleTestModel(device="cpu")

        # Test pruning to layer2
        pruned_model = prune_model_to_layer(
            base_model=model, frozen=True, layers=["layer1", "layer2"]
        )

        # Check that the model is frozen
        assert pruned_model.training is False
        for param in pruned_model.parameters():
            assert param.requires_grad is False

        # Check that layer3 and final_layer are not in the pruned model
        assert not hasattr(pruned_model, "layer3")
        assert not hasattr(pruned_model, "final_layer")

        # Check that layer1 and layer2 are preserved
        assert hasattr(pruned_model, "layer1")
        assert hasattr(pruned_model, "layer2")

    def test_prune_model_to_layer_unfrozen(self) -> None:
        """Test model pruning with unfrozen state."""
        model = SimpleTestModel(device="cpu")

        # Test pruning to layer3 with unfrozen state
        pruned_model = prune_model_to_layer(
            base_model=model,
            frozen=False,
            layers=["layer1", "layer2", "layer3"],
        )

        # Check that the model is in training mode
        assert pruned_model.training is True
        for param in pruned_model.parameters():
            assert param.requires_grad is True

        # Check that final_layer is not in the pruned model
        assert not hasattr(pruned_model, "final_layer")

        # Check that layer1, layer2, and layer3 are preserved
        assert hasattr(pruned_model, "layer1")
        assert hasattr(pruned_model, "layer2")
        assert hasattr(pruned_model, "layer3")

    def test_prune_model_to_layer_uppermost_detection(self) -> None:
        """Test that the uppermost layer is correctly identified."""
        model = SimpleTestModel(device="cpu")

        # Test with layers at different depths
        pruned_model = prune_model_to_layer(
            base_model=model,
            frozen=True,
            layers=["layer1", "layer2", "layer3"],
        )

        # layer3 should be the uppermost (deepest) layer
        # Check that final_layer is not in the pruned model
        assert not hasattr(pruned_model, "final_layer")

        # Check that layer1, layer2, and layer3 are preserved
        assert hasattr(pruned_model, "layer1")
        assert hasattr(pruned_model, "layer2")
        assert hasattr(pruned_model, "layer3")

    def test_prune_model_to_layer_empty_layers(self) -> None:
        """Test that empty layers list raises ValueError."""
        model = SimpleTestModel(device="cpu")

        with pytest.raises(ValueError, match="Layers list cannot be empty"):
            prune_model_to_layer(base_model=model, frozen=True, layers=[])

    def test_prune_model_to_layer_invalid_layers(self) -> None:
        """Test that invalid layer names raise ValueError."""
        model = SimpleTestModel(device="cpu")

        with pytest.raises(ValueError, match="None of the specified layers"):
            prune_model_to_layer(
                base_model=model,
                frozen=True,
                layers=["nonexistent_layer1", "nonexistent_layer2"],
            )

    def test_prune_model_to_layer_single_layer(self) -> None:
        """Test pruning to a single layer."""
        model = SimpleTestModel(device="cpu")

        pruned_model = prune_model_to_layer(
            base_model=model, frozen=False, layers=["layer1"]
        )

        # Check that only layer1 is preserved
        assert hasattr(pruned_model, "layer1")
        assert not hasattr(pruned_model, "layer2")
        assert not hasattr(pruned_model, "layer3")
        assert not hasattr(pruned_model, "final_layer")

        # Check that the model is in training mode
        assert pruned_model.training is True
        for param in pruned_model.parameters():
            assert param.requires_grad is True

    def test_prune_model_to_layer_forward_pass(self) -> None:
        """Test that the pruned model can perform forward pass."""
        model = SimpleTestModel(device="cpu")

        pruned_model = prune_model_to_layer(
            base_model=model, frozen=True, layers=["layer1", "layer2"]
        )

        # Create test input
        x = torch.randn(2, 10)  # batch_size=2, input_dim=10

        # Should be able to forward pass through pruned layers
        with torch.no_grad():
            output = pruned_model.layer1(x)
            output = pruned_model.layer2(output)

        # Check output shape (should be 15 based on layer2 output)
        assert output.shape == (2, 15)
