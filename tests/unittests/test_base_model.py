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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
    layers = ["model.0", "model.1"]  # Sequential layers are named 0 and 1
    embeddings = model.extract_embeddings(x, layers)

    # Check output shape
    expected_shape = (batch_size, 20 + 30)  # Concatenated embeddings from both layers
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
    layers = ["model.0", "model.1"]  # Sequential layers are named 0 and 1
    embeddings = model.extract_embeddings(x, layers)

    # Check output shape
    expected_shape = (batch_size, 20 + 30)  # Concatenated embeddings from both layers
    assert embeddings.shape == expected_shape
    assert embeddings.device == device


def test_extract_embeddings_invalid_layers() -> None:
    """Test extract_embeddings with invalid layer names."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_inference()

    # Create dummy input
    x = torch.randn(2, 10).to(device)

    # Try to extract embeddings from non-existent layer
    with pytest.raises(ValueError, match="No layers found matching"):
        model.extract_embeddings(x, ["nonexistent_layer"])


def test_extract_embeddings_gradient_propagation() -> None:
    """Test that gradients can propagate through extract_embeddings."""
    device = torch.device("cpu")
    model = MockModel(device)
    model.prepare_train()  # Set to training mode

    # Create dummy input
    x = torch.randn(2, 10).to(device)
    x.requires_grad = True

    # Extract embeddings
    layers = ["model.0", "model.1"]  # Sequential layers are named 0 and 1
    embeddings = model.extract_embeddings(x, layers)

    # Compute loss and backpropagate
    loss = embeddings.sum()
    loss.backward()

    # Check that gradients were computed
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


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
    layers = ["model.0", "model.1"]
    embeddings = model.extract_embeddings(x, layers)

    print("\nExtract Embeddings Test Results:")
    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Device: {embeddings.device}")
    print(f"Layer names: {layers}")
    print("Number of features per layer: [20, 30]")
    print(f"Total features: {embeddings.shape[1]}")


if __name__ == "__main__":
    test_extract_embeddings_main()
