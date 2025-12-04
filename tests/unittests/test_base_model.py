from typing import Any, Dict

import pytest
import torch

from representation_learning.models.base_model import ModelBase


class MockModel(ModelBase):
    """Mock model for testing extract_embeddings functionality."""

    def __init__(self, device: str, audio_config: Dict[str, Any] = None) -> None:
        super().__init__(device, audio_config)
        # Create a proper model structure
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.Linear(20, 30))

        # Discover layers for hook management
        self._hook_layers = ["model.0", "model.1"]

        # Discover linear layers for 'all' functionality
        self._discover_linear_layers()

        # Register hooks for the discovered layers
        self.register_hooks_for_layers(self._hook_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x)

    def _create_hook_fn(self, layer_name: str) -> callable:
        """Create a hook function that produces 3D embeddings for testing aggregation.

        Returns:
            A hook function that returns 3D embeddings
        """

        def hook_fn(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Convert 2D output to 3D (batch, seq_len, features) for testing
            if output.dim() == 2:
                # Add a sequence dimension
                output_3d = output.unsqueeze(1)  # (batch, 1, features)
                self._hook_outputs[layer_name] = output_3d
            else:
                self._hook_outputs[layer_name] = output

        return hook_fn

    def prepare_inference(self) -> None:
        """Override to handle our direct layer structure."""
        self.eval()
        self.to(self.device)

    def prepare_train(self) -> None:
        """Override to handle our direct layer structure."""
        self.train()
        self.to(self.device)


class TestExtractEmbeddings:
    """Test extract_embeddings functionality."""

    @pytest.fixture(scope="class")
    def model(self) -> MockModel:
        """Create a mock model for testing.

        Model is created once per test class to improve performance.

        Returns:
            MockModel: A configured mock model for testing.
        """
        device = torch.device("cpu")
        model = MockModel(device)
        model.prepare_inference()
        return model

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
            original_hook_layers = ["model.0", "model.1"]
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

    def test_extract_embeddings_input_formats(self, model: MockModel, sample_input: torch.Tensor) -> None:
        """Test extraction with tensor and dict inputs."""
        # Tensor input
        embeddings_tensor = model.extract_embeddings(sample_input, aggregation="mean")

        # Dict input
        dict_input = {
            "raw_wav": torch.randn(2, 10),
            "padding_mask": torch.ones(2, 10),
        }
        embeddings_dict = model.extract_embeddings(dict_input, aggregation="mean")

        # Both should work
        expected_shape = (2, 50)  # 20 + 30 features
        assert embeddings_tensor.shape == expected_shape
        assert embeddings_dict.shape == expected_shape
        assert embeddings_tensor.device == model.device
        assert embeddings_dict.device == model.device

    def test_extract_embeddings_aggregation_modes(self, model: MockModel, sample_input: torch.Tensor) -> None:
        """Test different aggregation modes."""
        # Mean aggregation
        embeddings_mean = model.extract_embeddings(sample_input, aggregation="mean")

        # None aggregation
        embeddings_none = model.extract_embeddings(sample_input, aggregation="none")

        # Check mean aggregation
        assert embeddings_mean.shape == (2, 50)
        assert embeddings_mean.device == model.device

        # Check none aggregation
        assert isinstance(embeddings_none, list)
        assert len(embeddings_none) == 2
        assert embeddings_none[0].shape == (2, 1, 20)
        assert embeddings_none[1].shape == (2, 1, 30)

    def test_extract_embeddings_gradient_propagation(self, model: MockModel) -> None:
        """Test that gradients can propagate through extract_embeddings."""
        model.prepare_train()
        try:
            x = torch.randn(2, 10, requires_grad=True)

            embeddings = model.extract_embeddings(x, aggregation="mean")
            loss = embeddings.sum()
            loss.backward()

            assert x.grad is not None
            assert not torch.isnan(x.grad).any()
        finally:
            # Reset model to eval mode to maintain test isolation
            model.prepare_inference()

    def test_extract_embeddings_error_handling(self, model: MockModel, sample_input: torch.Tensor) -> None:
        """Test error handling for invalid layer configurations."""
        # Remove hooks
        model.deregister_all_hooks()
        assert len(model._hooks) == 0

        # Should fallback to main features when no hooks
        result = model.extract_embeddings(sample_input)
        assert result is not None

    def test_get_last_non_classification_layer(self, model: MockModel) -> None:
        """Test that _get_last_non_classification_layer returns the correct layer."""
        last_layer = model._get_last_non_classification_layer()
        assert last_layer == "model.1"

        # Test with empty layer names
        model._layer_names = []
        last_layer = model._get_last_non_classification_layer()
        assert last_layer is None
