from typing import Optional

import pytest
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase


class SimpleTestModel(ModelBase):
    """A simple test model with multiple linear layers for testing."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)

        # Create a simple model with multiple linear layers
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 15)
        self.linear3 = nn.Linear(15, 8)
        self.final_layer = nn.Linear(8, 5)

        # Add some non-linear layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Simple forward pass through all layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.final_layer(x)
        return x


class TestExtractEmbeddingsAllLayers:
    """Test the 'all' functionality for extracting embeddings from all linear layers."""

    def test_extract_all_linear_layers(self) -> None:
        """Test that 'all' correctly finds and extracts from all linear layers."""
        model = SimpleTestModel(device="cpu")

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using 'all'
        embeddings = model.extract_embeddings(
            x=input_tensor, layers=["all"], average_over_time=True
        )

        # Should have embeddings from 3 linear layers (linear1, linear2, linear3)
        # final_layer is excluded as it's the classification layer
        expected_features = 20 + 15 + 8  # Sum of first 3 linear layer output dimensions
        assert embeddings.shape == (batch_size, expected_features)

    def test_extract_specific_and_all_layers(self) -> None:
        """Test that 'all' works in combination with specific layer names."""
        model = SimpleTestModel(device="cpu")

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using both 'all' and a specific layer
        embeddings = model.extract_embeddings(
            x=input_tensor,
            layers=["all", "linear1"],  # 'linear1' will be included once (deduplicated)
            average_over_time=True,
        )

        # Should have embeddings from 3 linear layers (linear1 appears once)
        # final_layer is excluded as it's the classification layer
        expected_features = (
            20 + 15 + 8
        )  # Sum of first 3 linear layer outputs (no duplication)
        assert embeddings.shape == (batch_size, expected_features)

    def test_no_linear_layers_found(self) -> None:
        """Test behavior when no linear layers are found."""

        # Create a model with no linear layers
        class NoLinearModel(ModelBase):
            def __init__(self, device: str = "cpu") -> None:
                super().__init__(device=device)
                self.conv = nn.Conv1d(1, 1, 3)
                self.relu = nn.ReLU()

            def forward(
                self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                x = x.unsqueeze(1)  # Add channel dimension
                x = self.conv(x)
                x = self.relu(x)
                return x.squeeze(1)

        model = NoLinearModel(device="cpu")
        input_tensor = torch.randn(2, 10)

        # Should raise ValueError when no layers are found
        with pytest.raises(ValueError, match="No layers found matching"):
            model.extract_embeddings(
                x=input_tensor, layers=["all"], average_over_time=True
            )

    def test_extract_without_averaging(self) -> None:
        """Test that 'all' works correctly when average_over_time=False."""
        model = SimpleTestModel(device="cpu")

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings without averaging
        embeddings = model.extract_embeddings(
            x=input_tensor, layers=["all"], average_over_time=False
        )

        # Should return a list of embeddings
        assert isinstance(embeddings, list)
        assert (
            len(embeddings) == 3
        )  # 3 linear layers (excluding final classification layer)

        # Check that each embedding has the correct shape
        expected_shapes = [(batch_size, 20), (batch_size, 15), (batch_size, 8)]
        for emb, expected_shape in zip(embeddings, expected_shapes, strict=False):
            assert emb.shape == expected_shape

    def test_classification_layer_excluded(self) -> None:
        """Test that the final classification layer is properly excluded."""
        model = SimpleTestModel(device="cpu")

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using 'all'
        embeddings = model.extract_embeddings(
            x=input_tensor, layers=["all"], average_over_time=True
        )

        # Verify that final_layer (classification layer) is not included
        # The embeddings should only come from linear1, linear2, linear3
        expected_features = 20 + 15 + 8  # 43 features total
        assert embeddings.shape == (batch_size, expected_features)

        # Test that explicitly including final_layer works
        embeddings_with_final = model.extract_embeddings(
            x=input_tensor,
            layers=[
                "all",
                "final_layer",
            ],  # Explicitly include the classification layer
            average_over_time=True,
        )

        # Should now include the final layer
        expected_features_with_final = 20 + 15 + 8 + 5  # 48 features total
        assert embeddings_with_final.shape == (batch_size, expected_features_with_final)
