from typing import Optional

import pytest
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase


class MockTestModel(ModelBase):
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

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    @pytest.fixture(scope="class")
    def model(self) -> MockTestModel:
        """Create a test model for testing.

        Model is created once per test class to improve performance.

        Returns:
            MockTestModel: A configured test model for testing.
        """
        return MockTestModel()

    @pytest.fixture(autouse=True)
    def setup_and_cleanup_hooks(self, request: pytest.FixtureRequest) -> None:
        """Ensure hooks are set up before and cleaned up after each test.

        Args:
            request: Pytest request object to access test fixtures.

        Yields:
            None: Yields control to the test, then cleans up hooks after.
        """
        if "model" in request.fixturenames:
            model = request.getfixturevalue("model")
            # Always re-register original hooks at start of test to ensure consistent state
            # (previous tests may have modified hooks)
            original_hook_layers = ["layer1", "layer2", "output"]
            model.register_hooks_for_layers(original_hook_layers)
        yield
        if "model" in request.fixturenames:
            model = request.getfixturevalue("model")
            model.deregister_all_hooks()

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Create sample input tensor.

        Returns:
            torch.Tensor: Random input tensor with shape (2, 10).
        """
        return torch.randn(2, 10)

    def test_model_initialization_and_layer_discovery(self, model: MockTestModel) -> None:
        """Test model initialization, hook setup, and layer discovery."""
        # Check hook storage
        assert isinstance(model._hooks, dict)
        assert isinstance(model._hook_outputs, dict)
        assert len(model._hooks) == 3
        assert len(model._hook_layers) == 3

        # Check layer discovery
        assert len(model._layer_names) == 3
        assert "layer1" in model._layer_names
        assert "layer2" in model._layer_names
        assert "output" in model._layer_names

        # Test getting linear layers excluding last
        linear_layers = model._get_linear_layers_excluding_last()
        assert len(linear_layers) == 2
        assert "layer1" in linear_layers
        assert "layer2" in linear_layers
        assert "output" not in linear_layers

    def test_hook_registration_and_management(self, model: MockTestModel) -> None:
        """Test hook registration and re-registration."""
        # Initially hooks are registered from __init__
        assert len(model._hooks) == 3

        # Register hooks for different layers
        model.register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2
        assert "layer1" in model._hooks
        assert "layer2" in model._hooks

        # Register same layers again
        model.register_hooks_for_layers(["layer1", "layer2"])
        assert len(model._hooks) == 2

    def test_hook_output_capture_and_clearing(self, model: MockTestModel, sample_input: torch.Tensor) -> None:
        """Test hook output capture and clearing."""
        # Forward pass
        model(sample_input)

        # Check outputs are captured
        assert "layer1" in model._hook_outputs
        assert "layer2" in model._hook_outputs
        assert model._hook_outputs["layer1"].shape == (2, 20)
        assert model._hook_outputs["layer2"].shape == (2, 30)

        # Clear outputs
        model._clear_hook_outputs()
        assert len(model._hook_outputs) == 0

        # Hooks should still be registered
        assert len(model._hooks) == 3

    def test_extract_embeddings_functionality(self, model: MockTestModel, sample_input: torch.Tensor) -> None:
        """Test extract_embeddings with hooks and consistency."""
        # First call
        result1 = model.extract_embeddings(sample_input, aggregation="mean")

        # Second call
        result2 = model.extract_embeddings(sample_input, aggregation="mean")

        # Results should be consistent
        assert torch.allclose(result1, result2)
        assert result1.shape[1] == 55  # 20 + 30 + 5 dimensions

        # Hooks should still be registered
        assert len(model._hooks) == 3

    def test_hook_persistence_and_cleanup(self, model: MockTestModel, sample_input: torch.Tensor) -> None:
        """Test hook persistence across forward passes and cleanup."""
        initial_hook_count = len(model._hooks)
        assert initial_hook_count == 3

        # Multiple forward passes
        for _ in range(3):
            model(sample_input)
            assert len(model._hooks) == initial_hook_count

        # Cleanup hooks
        model.deregister_all_hooks()
        assert len(model._hooks) == 0
        assert len(model._hook_outputs) == 0

        # Test fallback when no hooks
        result = model.extract_embeddings(sample_input)
        assert result is not None
