"""Tests for WeightedLinearProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.weighted_linear_probe import (
    WeightedLinearProbe,
)


class MockAudioProcessor:
    """Mock audio processor for testing."""

    def __init__(self, target_length: int = 1000, sr: int = 16000) -> None:
        self.target_length = target_length
        self.sr = sr
        self.target_length_seconds = target_length / sr


class MockBaseModel(ModelBase):
    """Mock base model for testing."""

    def __init__(self, embedding_dims: list, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.embedding_dims = embedding_dims
        self.audio_processor = MockAudioProcessor()
        self._hooks = {}

    def extract_embeddings(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        aggregation: str = "mean",
    ) -> torch.Tensor | list[torch.Tensor]:
        """Mock extract_embeddings method.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Mock embeddings tensor or list of tensors.
        """
        batch_size = x.shape[0]

        if aggregation == "none":
            # Return list of embeddings with different dimensions
            embeddings = []
            for dim in self.embedding_dims:
                # Create 2D tensor: (batch_size, embedding_dim)
                emb = torch.randn(batch_size, dim, device=self.device)
                embeddings.append(emb)
            return embeddings
        else:
            # Return single tensor
            emb_dim = self.embedding_dims[0] if self.embedding_dims else 256
            return torch.randn(batch_size, emb_dim, device=self.device)

    def register_hooks_for_layers(self, layers: list) -> None:
        """Mock register_hooks_for_layers method."""
        self._hooks = {layer: None for layer in layers}

    def deregister_all_hooks(self) -> None:
        """Mock register_hooks_for_layers method."""
        self._hooks.clear()


class TestWeightedLinearProbe:
    """Test cases for WeightedLinearProbe."""

    def test_feature_mode_with_input_dim(self) -> None:
        """Test WeightedLinearProbe in feature mode with provided input_dim."""
        input_dim = 512
        num_classes = 10
        batch_size = 4

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
        )

        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "classifier")
        assert not hasattr(probe, "layer_weights")

    def test_feature_mode_with_base_model(self) -> None:
        """Test WeightedLinearProbe in feature mode with base_model."""
        embedding_dims = [256]
        num_classes = 5
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
        )

        # Test forward pass
        x = torch.randn(batch_size, embedding_dims[0])
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "classifier")
        assert not hasattr(probe, "layer_weights")

    def test_single_tensor_case(self) -> None:
        """Test WeightedLinearProbe with single tensor embeddings."""
        embedding_dims = [256]
        num_classes = 8
        batch_size = 3

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "classifier")
        assert not hasattr(probe, "layer_weights")

    def test_list_embeddings_same_dimensions(self) -> None:
        """Test WeightedLinearProbe with list of embeddings having same dimensions."""
        embedding_dims = [256, 256, 256]  # Same dimensions
        num_classes = 6
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "classifier")
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (len(embedding_dims),)

        # Test that weights are learnable parameters
        assert probe.layer_weights.requires_grad is True

        # Test debug info
        debug_info = probe.debug_info()
        assert debug_info["probe_type"] == "weighted_linear"
        assert debug_info["has_layer_weights"] is True
        assert len(debug_info["layer_weights"]) == len(embedding_dims)

    def test_list_embeddings_different_dimensions_raises_error(self) -> None:
        """Test that WeightedLinearProbe raises error for different embedding
        dimensions."""
        embedding_dims = [256, 512, 128]  # Different dimensions

        base_model = MockBaseModel(embedding_dims)

        with pytest.raises(
            ValueError, match="All embeddings must have the same dimension"
        ):
            WeightedLinearProbe(
                base_model=base_model,
                layers=["layer1", "layer2", "layer3"],
                num_classes=5,
                device="cpu",
                feature_mode=False,
                aggregation="none",
            )

    def test_feature_mode_without_input_dim_raises_error(self) -> None:
        """Test that WeightedLinearProbe raises error in feature mode without
        input_dim."""
        with pytest.raises(
            ValueError, match="input_dim must be provided when feature_mode=True"
        ):
            WeightedLinearProbe(
                base_model=None,
                layers=[],
                num_classes=5,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_projection_dim_parameter(self) -> None:
        """Test WeightedLinearProbe with projection_dim parameter."""
        input_dim = 128
        num_classes = 4
        batch_size = 2
        projection_dim = 64

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            projection_dim=projection_dim,
        )

        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.projection_dim == projection_dim

    def test_freeze_backbone(self) -> None:
        """Test WeightedLinearProbe with frozen backbone."""
        embedding_dims = [256]
        num_classes = 3
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            freeze_backbone=True,
        )

        assert probe.freeze_backbone is True

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_dict_input(self) -> None:
        """Test WeightedLinearProbe with dictionary input."""
        input_dim = 64
        num_classes = 2
        batch_size = 2

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
        )

        # Test with dictionary input
        x = {
            "raw_wav": torch.randn(batch_size, input_dim),
            "padding_mask": torch.ones(batch_size, input_dim, dtype=torch.bool),
        }
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_weighted_sum_behavior(self) -> None:
        """Test that weighted sum is applied correctly for list embeddings."""
        embedding_dims = [128, 128, 128]
        num_classes = 3
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        # Manually set weights to test behavior
        with torch.no_grad():
            probe.layer_weights[0] = 1.0
            probe.layer_weights[1] = 0.0
            probe.layer_weights[2] = 0.0

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

        # Check that weights are normalized (softmax applied)
        weights = torch.softmax(probe.layer_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        input_dim = 32
        num_classes = 2

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
        )

        debug_info = probe.debug_info()

        expected_keys = [
            "probe_type",
            "layers",
            "feature_mode",
            "aggregation",
            "freeze_backbone",
            "target_length",
            "projection_dim",
            "has_layer_weights",
        ]

        for key in expected_keys:
            assert key in debug_info

        assert debug_info["probe_type"] == "weighted_linear"
        assert debug_info["feature_mode"] is True
        assert debug_info["has_layer_weights"] is False

    def test_cleanup_hooks(self) -> None:
        """Test that hooks are properly cleaned up."""
        embedding_dims = [128]

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=2,
            device="cpu",
            feature_mode=False,
        )

        # Check that hooks are registered
        assert len(base_model._hooks) == 1

        # Cleanup
        del probe

        # Check that hooks are cleaned up
        assert len(base_model._hooks) == 0

    def test_print_learned_weights_with_weights(self) -> None:
        """Test print_learned_weights method when weights exist."""
        embedding_dims = [128, 128, 128]
        layers = ["layer1", "layer2", "layer3"]

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLinearProbe(
            base_model=base_model,
            layers=layers,
            num_classes=3,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        # Manually set some weights to test output
        with torch.no_grad():
            probe.layer_weights[0] = 1.0
            probe.layer_weights[1] = 2.0
            probe.layer_weights[2] = 0.5

        # Capture print output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            probe.print_learned_weights()
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Check that output contains expected information
        assert "Learned Layer Weights:" in output
        assert "layer1" in output
        assert "layer2" in output
        assert "layer3" in output
        assert "Raw Weight" in output
        assert "Normalized" in output
        assert "Percentage" in output
        assert "Sum of normalized weights:" in output
        assert "Number of layers: 3" in output

    def test_print_learned_weights_without_weights(self) -> None:
        """Test print_learned_weights method when no weights exist."""
        input_dim = 64
        num_classes = 2

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
        )

        # Capture print output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            probe.print_learned_weights()
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Check that output contains expected message
        assert "No learned weights found" in output
        assert "does not use weighted sum" in output

    def test_2d_embedding_requirement(self) -> None:
        """Test that WeightedLinearProbe requires 2D embeddings."""

        # Create a mock that returns 3D embeddings (should fail)
        class MockBaseModel3D(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return 3D embeddings (should cause error)
                    return [torch.randn(batch_size, 10, 128, device=self.device)]
                else:
                    return torch.randn(batch_size, 10, 128, device=self.device)

        base_model = MockBaseModel3D([128])

        with pytest.raises(ValueError, match="Linear probe expects 2D embeddings"):
            WeightedLinearProbe(
                base_model=base_model,
                layers=["layer1"],
                num_classes=5,
                device="cpu",
                feature_mode=False,
                aggregation="mean",
            )

    def test_target_length_parameter(self) -> None:
        """Test WeightedLinearProbe with target_length parameter."""
        input_dim = 96
        num_classes = 3
        batch_size = 2
        target_length = 2000

        probe = WeightedLinearProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            target_length=target_length,
        )

        assert probe.target_length == target_length

        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
