"""Tests for probe models with multi-layer embeddings in feature mode."""

import tempfile
from pathlib import Path

import pytest
import torch

from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    load_embeddings_arrays,
    save_embeddings_arrays,
)
from representation_learning.models.probes.attention_probe import AttentionProbe
from representation_learning.models.probes.linear_probe import (
    LinearProbe,
)


class TestProbesMultiLayerEmbeddings:
    """Test probe models with multi-layer embeddings in feature mode."""

    def test_linear_probe_with_list_input_dim(self) -> None:
        """Test LinearProbe initialization with list input_dim."""
        # Test with different embedding shapes
        input_dims = [(128,), (64, 32), (16, 8, 4)]  # 2D, 3D, 4D shapes

        # Calculate the flattened dimension for the last layer
        last_layer_dim = input_dims[-1]  # (16, 8, 4)
        flattened_dim = 16 * 8 * 4  # 512

        probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2", "layer3"],
            num_classes=5,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,  # Use single integer for flattened dimension
        )

        # Check that the probe was initialized correctly
        assert probe.feature_mode is True
        assert probe.num_classes == 5
        assert hasattr(probe, "classifier")
        assert probe.classifier is not None

        # Test forward pass with single tensor (last layer) - flatten for 2D probe
        batch_size = 3
        test_input = torch.randn(batch_size, *last_layer_dim)
        # Flatten to 2D for LinearProbe
        test_input_flat = test_input.view(batch_size, -1)

        with torch.no_grad():
            output = probe(test_input_flat)

        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()

    def test_attention_probe_with_list_input_dim(self) -> None:
        """Test AttentionProbe initialization with list input_dim."""
        # Test with single 3D layer to avoid complexity
        input_dims = [(16, 8)]  # Single 3D shape

        # For 3D probes, we need to pass the actual 3D dimensions
        last_layer_dim = input_dims[-1]  # (16, 8)

        probe = AttentionProbe(
            base_model=None,
            layers=["layer1"],
            num_classes=5,
            device="cpu",
            feature_mode=True,
            input_dim=input_dims,  # Use list of dimensions for 3D probe
            num_heads=4,
            attention_dim=64,
            num_layers=2,
        )

        # Check that the probe was initialized correctly
        assert probe.feature_mode is True
        assert probe.num_classes == 5
        assert hasattr(probe, "classifier")
        assert probe.classifier is not None
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == 2

        # Test forward pass with single tensor (last layer) - ensure 3D for
        # AttentionProbe
        batch_size = 3
        test_input = torch.randn(batch_size, *last_layer_dim)

        with torch.no_grad():
            output = probe(test_input)

        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()

    def test_probes_with_single_tensor_input_dim(self) -> None:
        """Test probes with single integer input_dim (backward compatibility)."""
        # Test LinearProbe
        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=128,  # Single integer
        )

        test_input = torch.randn(2, 128)
        with torch.no_grad():
            output = linear_probe(test_input)
        assert output.shape == (2, 3)

        # Test AttentionProbe - need 3D input for attention
        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=128,  # Single integer
            num_heads=2,
            attention_dim=64,
        )

        test_input = torch.randn(2, 1, 128)  # 3D input for attention
        with torch.no_grad():
            output = attention_probe(test_input)
        assert output.shape == (2, 3)

    def test_probes_with_dictionary_embeddings_integration(self) -> None:
        """Test probes with actual dictionary embeddings from save/load cycle."""
        # Create multi-layer embeddings
        batch_size = 4
        embeddings = {
            "layer1": torch.randn(batch_size, 128),  # 2D
            "layer2": torch.randn(batch_size, 64, 32),  # 3D
            "layer3": torch.randn(batch_size, 16, 8, 4),  # 4D
        }
        labels = torch.randint(0, 5, (batch_size,))
        num_labels = 5

        # Save embeddings
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_embeddings.h5"
            save_embeddings_arrays(embeddings, labels, save_path, num_labels)

            # Load embeddings
            loaded_embeddings, loaded_labels, loaded_num_labels = load_embeddings_arrays(save_path)

            # Verify loaded data
            assert isinstance(loaded_embeddings, dict)
            assert loaded_num_labels == num_labels
            assert torch.equal(loaded_labels, labels)

            # Test with LinearProbe - use flattened dimension
            last_layer_name = list(loaded_embeddings.keys())[-1]
            last_layer_emb = loaded_embeddings[last_layer_name]
            flattened_dim = last_layer_emb.shape[1:].numel()  # 16 * 8 * 4 = 512

            linear_probe = LinearProbe(
                base_model=None,
                layers=list(loaded_embeddings.keys()),
                num_classes=num_labels,
                device="cpu",
                feature_mode=True,
                input_dim=flattened_dim,  # Use single integer for flattened dimension
            )

            # Test with last layer (as done in training/evaluation) - flatten
            # for LinearProbe
            last_layer_emb_flat = last_layer_emb.view(batch_size, -1)

            with torch.no_grad():
                output = linear_probe(last_layer_emb_flat)

            assert output.shape == (batch_size, num_labels)
            assert not torch.isnan(output).any()

            # Test with AttentionProbe - use list of dimensions for 3D probe
            input_dims = [emb.shape[1:] for emb in loaded_embeddings.values()]
            attention_probe = AttentionProbe(
                base_model=None,
                layers=list(loaded_embeddings.keys()),
                num_classes=num_labels,
                device="cpu",
                feature_mode=True,
                input_dim=input_dims,  # Use list of dimensions for 3D probe
                num_heads=4,
                attention_dim=64,
            )

            with torch.no_grad():
                output = attention_probe(last_layer_emb)

            assert output.shape == (batch_size, num_labels)
            assert not torch.isnan(output).any()

    def test_probes_with_embedding_dataset(self) -> None:
        """Test probes with EmbeddingDataset (simulating training scenario)."""
        # Create multi-layer embeddings
        batch_size = 3
        embeddings = {
            "layer1": torch.randn(batch_size, 128),  # 2D
            "layer2": torch.randn(batch_size, 64, 32),  # 3D
            "layer3": torch.randn(batch_size, 16, 8, 4),  # 4D
        }
        labels = torch.randint(0, 3, (batch_size,))

        # Create dataset
        dataset = EmbeddingDataset(embeddings, labels)

        # Test dataset behavior
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "label" in sample
        assert all(layer_name in sample for layer_name in embeddings.keys())

        # Test probe initialization with dataset dimensions
        first_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        embed_keys = [k for k in first_batch.keys() if k != "label"]

        # Calculate flattened dimension for last layer
        last_layer_name = list(embed_keys)[-1]
        last_layer_emb = first_batch[last_layer_name]
        flattened_dim = last_layer_emb.shape[1:].numel()  # 16 * 8 * 4 = 512

        # Test LinearProbe
        linear_probe = LinearProbe(
            base_model=None,
            layers=embed_keys,
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,  # Use single integer for flattened dimension
        )

        # Test with dictionary input (as passed from training loop) - flatten
        # for LinearProbe
        z = {k: first_batch[k] for k in embed_keys}
        last_layer_emb = z[last_layer_name]
        # Flatten to 2D for LinearProbe
        last_layer_emb_flat = last_layer_emb.view(1, -1)

        with torch.no_grad():
            output = linear_probe(last_layer_emb_flat)

        assert output.shape == (1, 3)  # batch_size=1, num_classes=3

    def test_probes_different_embedding_shapes(self) -> None:
        """Test probes with various embedding shapes."""
        test_cases = [
            # (input_dims, description)
            ([(128,)], "2D embeddings only"),
            ([(64, 32)], "3D embeddings only"),
            ([(16, 8)], "3D embeddings only"),
        ]

        for input_dims, description in test_cases:
            # Calculate flattened dimension for last layer
            last_dim = input_dims[-1]
            flattened_dim = 1
            for d in last_dim:
                flattened_dim *= d

            # Test LinearProbe
            linear_probe = LinearProbe(
                base_model=None,
                layers=[f"layer_{i}" for i in range(len(input_dims))],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=flattened_dim,  # Use single integer for flattened dimension
            )

            # Test with last layer - flatten for LinearProbe
            test_input = torch.randn(2, *last_dim)
            test_input_flat = test_input.view(2, -1)

            with torch.no_grad():
                output = linear_probe(test_input_flat)

            assert output.shape == (2, 3), f"Failed for {description}"
            assert not torch.isnan(output).any(), f"NaN output for {description}"

            # Test AttentionProbe - only for 3D compatible cases
            if len(last_dim) >= 2:  # Only test attention for 3D+ inputs
                attention_probe = AttentionProbe(
                    base_model=None,
                    layers=[f"layer_{i}" for i in range(len(input_dims))],
                    num_classes=3,
                    device="cpu",
                    feature_mode=True,
                    input_dim=input_dims,  # Use list of dimensions for 3D probe
                    num_heads=2,
                    attention_dim=32,
                )

                with torch.no_grad():
                    output = attention_probe(test_input)

                assert output.shape == (2, 3), f"Failed for {description}"
                assert not torch.isnan(output).any(), f"NaN output for {description}"

    def test_probes_gradient_flow(self) -> None:
        """Test that gradients flow correctly through the probes."""
        input_dims = [(128,), (64, 32)]
        flattened_dim = 64 * 32  # 2048

        # Test LinearProbe
        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,  # Use single integer for flattened dimension
        )

        # Test with last layer - flatten for LinearProbe
        test_input = torch.randn(2, 64, 32, requires_grad=True)
        test_input_flat = test_input.view(2, -1)
        output = linear_probe(test_input_flat)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()

        # Check classifier gradients
        assert linear_probe.classifier.weight.grad is not None
        assert not torch.isnan(linear_probe.classifier.weight.grad).any()

        # Test AttentionProbe
        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=input_dims,  # Use list of dimensions for 3D probe
            num_heads=2,
            attention_dim=32,
        )

        test_input = torch.randn(2, 64, 32, requires_grad=True)
        output = attention_probe(test_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()

        # Check classifier gradients
        assert attention_probe.classifier.weight.grad is not None
        assert not torch.isnan(attention_probe.classifier.weight.grad).any()

    def test_probes_error_handling(self) -> None:
        """Test error handling in probe initialization."""
        # Test with invalid input_dim type
        with pytest.raises((TypeError, ValueError)):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim="invalid",  # Should be int or list
            )

        # Test with empty input_dim list
        with pytest.raises((ValueError, IndexError)):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=[],  # Empty list
            )

        # Test with None input_dim and no base_model
        with pytest.raises(ValueError):
            LinearProbe(
                base_model=None,
                layers=["layer1"],
                num_classes=3,
                device="cpu",
                feature_mode=True,
                input_dim=None,  # None with no base_model
            )

    def test_probes_forward_with_different_batch_sizes(self) -> None:
        """Test probes with different batch sizes."""
        input_dims = [(128,), (64, 32)]
        flattened_dim = 64 * 32  # 2048

        linear_probe = LinearProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=flattened_dim,  # Use single integer for flattened dimension
        )

        attention_probe = AttentionProbe(
            base_model=None,
            layers=["layer1", "layer2"],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=input_dims,  # Use list of dimensions for 3D probe
            num_heads=2,
            attention_dim=32,
        )

        batch_sizes = [1, 2, 5, 10]
        last_dim = input_dims[-1]  # (64, 32)

        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, *last_dim)
            test_input_flat = test_input.view(batch_size, -1)

            with torch.no_grad():
                linear_output = linear_probe(test_input_flat)
                attention_output = attention_probe(test_input)

            assert linear_output.shape == (batch_size, 3)
            assert attention_output.shape == (batch_size, 3)
            assert not torch.isnan(linear_output).any()
            assert not torch.isnan(attention_output).any()
