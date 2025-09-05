"""Tests for WeightedAttentionProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.weighted_attention_probe import (
    WeightedAttentionProbe,
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
        super().__init__(device=device)
        self.device = device
        self.embedding_dims = embedding_dims
        self.audio_processor = MockAudioProcessor()
        self._hooks = {}

    def extract_embeddings(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
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
                # Create 3D tensor: (batch_size, sequence_length, embedding_dim)
                seq_len = 50  # Fixed sequence length for testing
                emb = torch.randn(batch_size, seq_len, dim, device=self.device)
                embeddings.append(emb)
            return embeddings
        else:
            # Return single tensor
            seq_len = 50
            emb_dim = self.embedding_dims[0] if self.embedding_dims else 256
            return torch.randn(batch_size, seq_len, emb_dim, device=self.device)

    def register_hooks_for_layers(self, layers: list) -> None:
        """Mock register_hooks_for_layers method."""
        self._hooks = {layer: None for layer in layers}

    def deregister_all_hooks(self) -> None:
        """Mock deregister_all_hooks method."""
        self._hooks.clear()


class TestWeightedAttentionProbe:
    """Test cases for WeightedAttentionProbe."""

    def test_feature_mode_with_input_dim(self) -> None:
        """Test WeightedAttentionProbe in feature mode with provided input_dim."""
        input_dim = 512
        num_classes = 10
        batch_size = 4
        seq_len = 50
        num_heads = 8
        attention_dim = 512
        num_layers = 2

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == num_layers
        assert not hasattr(probe, "layer_weights")

    def test_feature_mode_with_base_model(self) -> None:
        """Test WeightedAttentionProbe in feature mode with base_model."""
        embedding_dims = [256]
        num_classes = 5
        batch_size = 2
        seq_len = 50
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, seq_len, embedding_dims[0])
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "attention_layers")
        assert not hasattr(probe, "layer_weights")

    def test_single_tensor_case(self) -> None:
        """Test WeightedAttentionProbe with single tensor embeddings."""
        embedding_dims = [256]
        num_classes = 8
        batch_size = 3
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "attention_layers")
        assert not hasattr(probe, "layer_weights")

    def test_list_embeddings_same_dimensions(self) -> None:
        """Test WeightedAttentionProbe with list of embeddings having same
        dimensions."""
        embedding_dims = [256, 256, 256]  # Same dimensions
        num_classes = 6
        batch_size = 2
        num_heads = 4
        attention_dim = 256
        num_layers = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "attention_layers")
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (len(embedding_dims),)

        # Test that weights are learnable parameters
        assert probe.layer_weights.requires_grad is True

        # Test debug info
        debug_info = probe.debug_info()
        assert debug_info["probe_type"] == "weighted_attention"
        assert debug_info["has_layer_weights"] is True
        assert len(debug_info["layer_weights"]) == len(embedding_dims)

    def test_list_embeddings_different_dimensions_automatic_projection(self) -> None:
        """Test that WeightedAttentionProbe handles different embedding
        dimensions with automatic projection."""
        embedding_dims = [256, 512, 128]  # Different dimensions
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)

        # With automatic projection, different dimensions should work fine
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=5,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(2, 1000)
        output = probe(x)

        assert output.shape == (2, 5)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (3,)

    def test_feature_mode_without_input_dim_raises_error(self) -> None:
        """Test that WeightedAttentionProbe raises error in feature mode
        without input_dim."""
        with pytest.raises(
            ValueError, match="input_dim must be provided when feature_mode=True"
        ):
            WeightedAttentionProbe(
                base_model=None,
                layers=[],
                num_classes=5,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_positional_encoding(self) -> None:
        """Test WeightedAttentionProbe with positional encoding."""
        input_dim = 128
        num_classes = 4
        batch_size = 2
        seq_len = 30
        max_seq_len = 100
        num_heads = 4
        attention_dim = 128
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            use_positional_encoding=True,
            max_sequence_length=max_seq_len,
        )

        assert hasattr(probe, "pos_encoding")
        assert probe.pos_encoding.pe.shape == (1, max_seq_len, input_dim)

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_no_positional_encoding(self) -> None:
        """Test WeightedAttentionProbe without positional encoding."""
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            use_positional_encoding=False,
        )

        assert probe.pos_encoding is None

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_dropout_rate(self) -> None:
        """Test WeightedAttentionProbe with different dropout rates."""
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1
        dropout_rate = 0.3

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        assert probe.dropout_rate == dropout_rate

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_no_dropout(self) -> None:
        """Test WeightedAttentionProbe without dropout."""
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout_rate=0.0,
        )

        assert probe.dropout is None

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_multiple_attention_layers(self) -> None:
        """Test WeightedAttentionProbe with multiple attention layers."""
        input_dim = 96
        num_classes = 4
        batch_size = 2
        seq_len = 25
        num_heads = 6
        attention_dim = 96
        num_layers = 3
        dropout_rate = 0.2

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        # Check attention layers configuration
        assert len(probe.attention_layers) == num_layers

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_freeze_backbone(self) -> None:
        """Test WeightedAttentionProbe with frozen backbone."""
        embedding_dims = [256]
        num_classes = 3
        batch_size = 2
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            freeze_backbone=True,
        )

        assert probe.freeze_backbone is True

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_dict_input(self) -> None:
        """Test WeightedAttentionProbe with dictionary input."""
        input_dim = 64
        num_classes = 2
        batch_size = 2
        seq_len = 30
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test with dictionary input
        x = {
            "raw_wav": torch.randn(batch_size, seq_len, input_dim),
            "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_weighted_sum_behavior(self) -> None:
        """Test that weighted sum is applied correctly for list embeddings."""
        embedding_dims = [128, 128, 128]
        num_classes = 3
        batch_size = 2
        num_heads = 4
        attention_dim = 128
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
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

    def test_sequence_length_handling(self) -> None:
        """Test that different sequence lengths are handled correctly."""
        embedding_dims = [64, 64]  # Same dimensions, different sequence lengths
        num_classes = 2
        batch_size = 1
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        # Create base model that returns embeddings with different sequence lengths
        class MockBaseModelVariableSeq(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                if aggregation == "none":
                    # Return embeddings with different sequence lengths
                    emb1 = torch.randn(batch_size, 30, 64, device=self.device)
                    emb2 = torch.randn(batch_size, 40, 64, device=self.device)
                    return [emb1, emb2]
                else:
                    return torch.randn(batch_size, 35, 64, device=self.device)

        base_model = MockBaseModelVariableSeq(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        input_dim = 32
        num_classes = 2
        num_heads = 2
        attention_dim = 32
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        debug_info = probe.debug_info()

        expected_keys = [
            "probe_type",
            "layers",
            "feature_mode",
            "aggregation",
            "freeze_backbone",
            "num_heads",
            "attention_dim",
            "num_layers",
            "dropout_rate",
            "max_sequence_length",
            "use_positional_encoding",
            "target_length",
            "has_layer_weights",
        ]

        for key in expected_keys:
            assert key in debug_info

        assert debug_info["probe_type"] == "weighted_attention"
        assert debug_info["feature_mode"] is True
        assert debug_info["has_layer_weights"] is False
        assert debug_info["num_heads"] == num_heads
        assert debug_info["attention_dim"] == attention_dim
        assert debug_info["num_layers"] == num_layers

    def test_cleanup_hooks(self) -> None:
        """Test that hooks are properly cleaned up."""
        embedding_dims = [128]
        num_heads = 2
        attention_dim = 128
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=2,
            device="cpu",
            feature_mode=False,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Hooks are not registered in constructor anymore
        # They are registered in get_probe() function
        assert len(base_model._hooks) == 0

        # Cleanup
        del probe

        # Check that hooks are cleaned up (should still be 0)
        assert len(base_model._hooks) == 0

    def test_print_learned_weights_with_weights(self) -> None:
        """Test print_learned_weights method when weights exist."""
        embedding_dims = [128, 128, 128]
        layers = ["layer1", "layer2", "layer3"]
        num_heads = 4
        attention_dim = 128
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=layers,
            num_classes=3,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
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
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
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

    def test_2d_embedding_support(self) -> None:
        """Test that WeightedAttentionProbe now supports 2D embeddings."""

        # Create a mock that returns 2D embeddings (should work now)
        class MockBaseModel2D(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return 2D embeddings (should work now)
                    return [torch.randn(batch_size, 128, device=self.device)]
                else:
                    # For non-"none" aggregation, return 3D tensor as expected
                    # by attention
                    return torch.randn(batch_size, 10, 128, device=self.device)

        base_model = MockBaseModel2D([128])
        num_heads = 2
        attention_dim = 128
        num_layers = 1

        # This should now work without raising an error
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=5,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(2, 1000)
        output = probe(x)
        assert output.shape == (2, 5)

    def test_target_length_parameter(self) -> None:
        """Test WeightedAttentionProbe with target_length parameter."""
        input_dim = 96
        num_classes = 3
        batch_size = 2
        target_length = 2000
        num_heads = 2
        attention_dim = 96
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            target_length=target_length,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        assert probe.target_length == target_length

        # Test forward pass
        x = torch.randn(batch_size, 50, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_automatic_projection_mixed_embedding_shapes(self) -> None:
        """Test WeightedAttentionProbe with automatic projection for mixed
        embedding shapes."""

        # Create a mock that returns embeddings of different shapes (4D, 3D, 2D)
        class MockBaseModelMixedShapes(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return embeddings with different shapes
                    emb1 = torch.randn(
                        batch_size, 3, 4, 5, device=self.device
                    )  # 4D: (batch, 3, 4, 5)
                    emb2 = torch.randn(
                        batch_size, 10, 64, device=self.device
                    )  # 3D: (batch, 10, 64)
                    emb3 = torch.randn(
                        batch_size, 128, device=self.device
                    )  # 2D: (batch, 128)
                    return [emb1, emb2, emb3]
                else:
                    return torch.randn(batch_size, 10, 64, device=self.device)

        base_model = MockBaseModelMixedShapes([256, 256, 256])  # Dummy dims, not used
        num_classes = 5
        batch_size = 2
        num_heads = 4
        attention_dim = 128  # This will be the target feature dimension
        num_layers = 2

        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",  # Always use CPU for tests
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "attention_layers")
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (3,)  # Three embeddings

        # Check that embedding projector was created
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None

        # Test that weights are learnable parameters
        assert probe.layer_weights.requires_grad is True

        # Test debug info
        debug_info = probe.debug_info()
        assert debug_info["probe_type"] == "weighted_attention"
        assert debug_info["has_layer_weights"] is True
        assert len(debug_info["layer_weights"]) == 3

    def test_automatic_projection_4d_embeddings_only(self) -> None:
        """Test WeightedAttentionProbe with automatic projection for 4D
        embeddings only."""

        class MockBaseModel4D(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return multiple 4D embeddings with different dimensions
                    emb1 = torch.randn(
                        batch_size, 3, 4, 5, device=self.device
                    )  # (batch, 3, 4, 5)
                    emb2 = torch.randn(
                        batch_size, 2, 6, 8, device=self.device
                    )  # (batch, 2, 6, 8)
                    emb3 = torch.randn(
                        batch_size, 4, 3, 7, device=self.device
                    )  # (batch, 4, 3, 7)
                    return [emb1, emb2, emb3]
                else:
                    return torch.randn(batch_size, 10, 64, device=self.device)

        base_model = MockBaseModel4D([256, 256, 256])
        num_classes = 4
        batch_size = 2
        num_heads = 4
        attention_dim = 96  # Target feature dimension
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",  # Always use CPU for tests
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None

    def test_automatic_projection_3d_embeddings_only(self) -> None:
        """Test WeightedAttentionProbe with automatic projection for 3D
        embeddings only."""

        class MockBaseModel3D(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return multiple 3D embeddings with different dimensions
                    emb1 = torch.randn(
                        batch_size, 10, 64, device=self.device
                    )  # (batch, 10, 64)
                    emb2 = torch.randn(
                        batch_size, 20, 128, device=self.device
                    )  # (batch, 20, 128)
                    emb3 = torch.randn(
                        batch_size, 5, 32, device=self.device
                    )  # (batch, 5, 32)
                    return [emb1, emb2, emb3]
                else:
                    return torch.randn(batch_size, 10, 64, device=self.device)

        base_model = MockBaseModel3D([256, 256, 256])
        num_classes = 3
        batch_size = 2
        num_heads = 4
        attention_dim = 128  # Target feature dimension
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",  # Always use CPU for tests
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None

    def test_automatic_projection_2d_embeddings_only(self) -> None:
        """Test WeightedAttentionProbe with 2D embeddings that get converted to 3D."""

        class MockBaseModel2D(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return multiple 2D embeddings with DIFFERENT dimensions
                    # This will force projection to be created
                    emb1 = torch.randn(
                        batch_size, 32, device=self.device
                    )  # (batch, 32)
                    emb2 = torch.randn(
                        batch_size, 64, device=self.device
                    )  # (batch, 64)
                    emb3 = torch.randn(
                        batch_size, 128, device=self.device
                    )  # (batch, 128)
                    return [emb1, emb2, emb3]
                else:
                    return torch.randn(batch_size, 64, device=self.device)

        base_model = MockBaseModel2D([32, 64, 128])
        num_classes = 2
        batch_size = 2
        num_heads = 4
        attention_dim = 128  # Target feature dimension (max)
        num_layers = 1

        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",  # Always use CPU for tests
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None

    def test_automatic_projection_realistic_model_embeddings(self) -> None:
        """Test WeightedAttentionProbe with realistic model embeddings
        (EfficientNet, BEATs, BirdMAE)."""

        class MockBaseModelRealistic(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Simulate realistic model outputs
                    # EfficientNet: 4D convolutional features
                    efficientnet_emb = torch.randn(
                        batch_size, 1280, 7, 7, device=self.device
                    )  # (batch, 1280, 7, 7)

                    # BEATs/AVES: 3D sequence features
                    beats_emb = torch.randn(
                        batch_size, 100, 768, device=self.device
                    )  # (batch, 100, 768)

                    # BirdMAE: 2D global features
                    birdmae_emb = torch.randn(
                        batch_size, 1024, device=self.device
                    )  # (batch, 1024)

                    return [efficientnet_emb, beats_emb, birdmae_emb]
                else:
                    return torch.randn(batch_size, 100, 768, device=self.device)

        base_model = MockBaseModelRealistic([256, 256, 256])
        num_classes = 10
        batch_size = 2
        num_heads = 8
        attention_dim = 1024  # Target feature dimension (max of all embeddings)
        num_layers = 2

        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["efficientnet", "beats", "birdmae"],
            num_classes=num_classes,
            device="cpu",  # Always use CPU for tests
            feature_mode=False,
            aggregation="none",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None

        # Test that all embeddings are properly projected and combined
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (3,)  # Three model embeddings

        # Test debug info
        debug_info = probe.debug_info()
        assert debug_info["has_layer_weights"] is True
        assert debug_info["has_embedding_projectors"] is True

    def test_4d_embedding_support_efficientnet_style(self) -> None:
        """Test that WeightedAttentionProbe supports 4D embeddings like
        EfficientNet avgpool."""

        # Create a mock that returns 4D embeddings like EfficientNet avgpool
        class MockBaseModel4DEfficientNet(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return 4D embeddings like EfficientNet avgpool:
                    # (batch, 1280, 1, 1)
                    return [torch.randn(batch_size, 1280, 1, 1, device=self.device)]
                else:
                    return torch.randn(batch_size, 1280, 1, 1, device=self.device)

        base_model = MockBaseModel4DEfficientNet([1280])
        num_heads = 8
        attention_dim = 1280
        num_layers = 2

        # This should work without raising an error
        probe = WeightedAttentionProbe(
            base_model=base_model,
            layers=["model.avgpool"],
            num_classes=10,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        # Test forward pass
        x = torch.randn(2, 1000)
        output = probe(x)
        assert output.shape == (2, 10)

        # Test that the probe was initialized correctly
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == num_layers
        assert not hasattr(probe, "layer_weights")  # Single embedding case
