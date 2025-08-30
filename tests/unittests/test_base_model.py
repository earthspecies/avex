from typing import Any, Dict

import pytest
import torch

from representation_learning.models.base_model import ModelBase


class MockModel(ModelBase):
    """Mock model for testing extract_embeddings functionality."""

    def __init__(self, device: str, audio_config: Dict[str, Any] = None) -> None:
        super().__init__(device, audio_config)
        # Create a proper model structure
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.Linear(20, 30)
        )

        # Discover layers for hook management
        self._hook_layers = ["model.0", "model.1"]

        # Discover linear layers for 'all' functionality
        self._discover_linear_layers()

        # Register hooks for the discovered layers
        self.register_hooks_for_layers(self._hook_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _create_hook_fn(self, layer_name: str) -> callable:
        """Create a hook function that produces 3D embeddings for testing
        aggregation.

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


def test_extract_embeddings_basic() -> None:
    """Test basic functionality of extract_embeddings."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    batch_size = 2
    seq_length = 10
    x = torch.randn(batch_size, seq_length).to(device)

    # Extract embeddings from both layers
    embeddings = model.extract_embeddings(x, aggregation="mean")

    # Check output shape
    expected_shape = (
        batch_size,
        20 + 30,
    )  # Concatenated embeddings from both layers
    assert embeddings.shape == expected_shape
    assert embeddings.device == device


def test_extract_embeddings_dict_input() -> None:
    """Test extract_embeddings with dictionary input."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input dictionary
    batch_size = 2
    seq_length = 10
    x = {
        "raw_wav": torch.randn(batch_size, seq_length).to(device),
        "padding_mask": torch.ones(batch_size, seq_length).to(device),
    }

    # Extract embeddings from both layers
    embeddings = model.extract_embeddings(x, aggregation="mean")

    # Check output shape
    expected_shape = (
        batch_size,
        20 + 30,
    )  # Concatenated embeddings from both layers
    assert embeddings.shape == expected_shape
    assert embeddings.device == device


def test_extract_embeddings_invalid_layers() -> None:
    """Test extract_embeddings with no hooks registered."""
    device = torch.device("cpu")
    model = MockModel(device)
    # Remove hooks that were registered in __init__
    model.deregister_all_hooks()
    assert len(model._hooks) == 0

    # Create dummy input
    x = torch.randn(2, 10).to(device)

    # Try to extract embeddings without hooks registered
    with pytest.raises(ValueError, match="No hooks registered"):
        model.extract_embeddings(x)


def test_extract_embeddings_gradient_propagation() -> None:
    """Test that gradients can propagate through extract_embeddings."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_train()  # Set to training mode

    # Create dummy input
    x = torch.randn(2, 10).to(device)
    x.requires_grad = True

    # Extract embeddings
    embeddings = model.extract_embeddings(x, aggregation="mean")

    # Compute loss and backpropagate
    loss = embeddings.sum()
    loss.backward()

    # Check that gradients were computed
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_extract_embeddings_aggregation_mean() -> None:
    """Test extract_embeddings with aggregation='mean' (default)."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    batch_size = 2
    seq_length = 10
    x = torch.randn(batch_size, seq_length).to(device)

    # Extract embeddings with aggregation='mean' (default)
    embeddings = model.extract_embeddings(x, aggregation="mean")

    # Check output shape - should be concatenated
    expected_shape = (batch_size, 20 + 30)
    assert embeddings.shape == expected_shape
    assert embeddings.device == device


def test_extract_embeddings_aggregation_none() -> None:
    """Test extract_embeddings with aggregation='none'."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    batch_size = 2
    seq_length = 10
    x = torch.randn(batch_size, seq_length).to(device)

    # Extract embeddings with aggregation='none'
    embeddings = model.extract_embeddings(x, aggregation="none")

    # Check output - should be a list of individual embeddings
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2  # Two layers
    # First layer (3D: batch, seq, features)
    assert embeddings[0].shape == (batch_size, 1, 20)
    # Second layer (3D: batch, seq, features)
    assert embeddings[1].shape == (batch_size, 1, 30)
    assert embeddings[0].device == device
    assert embeddings[1].device == device


def test_extract_embeddings_single_layer_no_aggregation() -> None:
    """Test extract_embeddings with single layer (no aggregation needed)."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    batch_size = 2
    seq_length = 10
    x = torch.randn(batch_size, seq_length).to(device)

    # Extract embeddings from single layer
    embeddings = model.extract_embeddings(x, aggregation="mean")

    # Check output shape - should be from all registered hooks (both layers)
    expected_shape = (batch_size, 50)  # 20 + 30 features
    assert embeddings.shape == expected_shape


def test_extract_embeddings_main() -> None:
    """Main test function that demonstrates extract_embeddings functionality."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length).to(device)

    # Extract embeddings from both layers
    embeddings = model.extract_embeddings(x, aggregation="mean")

    print("\nExtract Embeddings Test Results:")
    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Device: {embeddings.device}")
    print(f"Layer names: {['model.0', 'model.1']}")
    print("Number of features per layer: [20, 30]")
    print(f"Total features: {embeddings.shape[1]}")


if __name__ == "__main__":
    test_extract_embeddings_main()
