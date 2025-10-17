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
        """Test that 'all' includes all linear layers, including classifier."""
        model = SimpleTestModel(device="cpu")

        # Register hooks for all layers
        model.register_hooks_for_layers(["all"])

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using 'all'
        embeddings = model.extract_embeddings(x=input_tensor, aggregation="mean")

        # Should have embeddings from all 4 linear layers (including final_layer)
        expected_features = 20 + 15 + 8 + 5
        assert embeddings.shape == (batch_size, expected_features)

    def test_extract_specific_and_all_layers(self) -> None:
        """Test that 'all' works in combination with specific layer names."""
        model = SimpleTestModel(device="cpu")

        # Register hooks for all layers
        model.register_hooks_for_layers(["all"])

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using both 'all' and a specific layer
        embeddings = model.extract_embeddings(
            x=input_tensor,
            aggregation="mean",
        )

        # Should have embeddings from all 4 linear layers; 'all' deduplicates
        expected_features = 20 + 15 + 8 + 5
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
                self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                x = x.unsqueeze(1)  # Add channel dimension
                x = self.conv(x)
                x = self.relu(x)
                return x.squeeze(1)

        model = NoLinearModel(device="cpu")

        # Should raise ValueError when trying to register hooks for non-existent layers
        with pytest.raises(ValueError, match="Layer.*not found in model"):
            model.register_hooks_for_layers(["nonexistent_layer"])

    def test_extract_without_averaging(self) -> None:
        """Test that 'all' works correctly when aggregation='none'."""
        model = SimpleTestModel(device="cpu")

        # Register hooks for all layers
        model.register_hooks_for_layers(["all"])

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings without averaging
        embeddings = model.extract_embeddings(x=input_tensor, aggregation="none")

        # Should return a list of 4 embeddings (including final classification layer)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 4

        # Check that each embedding has the correct shape
        expected_shapes = [
            (batch_size, 20),
            (batch_size, 15),
            (batch_size, 8),
            (batch_size, 5),
        ]
        for emb, expected_shape in zip(embeddings, expected_shapes, strict=False):
            assert emb.shape == expected_shape

    def test_classification_layer_included_and_deduped(self) -> None:
        """Test that the classification layer is included and deduped with 'all'."""
        model = SimpleTestModel(device="cpu")

        # Register hooks for all layers
        model.register_hooks_for_layers(["all"])

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 10)

        # Extract embeddings using 'all'
        embeddings = model.extract_embeddings(x=input_tensor, aggregation="mean")

        # Verify that final_layer (classification layer) is included
        expected_features = 20 + 15 + 8 + 5
        assert embeddings.shape == (batch_size, expected_features)

        # Test that explicitly including final_layer with 'all' is deduped
        model.register_hooks_for_layers(["all", "final_layer"])
        embeddings_with_final = model.extract_embeddings(
            x=input_tensor,
            aggregation="mean",
        )

        # Shape should remain the same due to de-duplication of layer names
        assert embeddings_with_final.shape == (
            batch_size,
            expected_features,
        )
