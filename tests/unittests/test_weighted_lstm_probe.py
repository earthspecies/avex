"""Tests for WeightedLSTMProbe."""

import pytest
import torch
import torch.nn as nn

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.weighted_lstm_probe import WeightedLSTMProbe


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


class TestWeightedLSTMProbe:
    """Test cases for WeightedLSTMProbe."""

    def test_feature_mode_with_input_dim(self) -> None:
        """Test WeightedLSTMProbe in feature mode with provided input_dim."""
        input_dim = 512
        num_classes = 10
        batch_size = 4
        seq_len = 50

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            lstm_hidden_size=128,
            num_layers=1,
            bidirectional=False,
        )

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "lstm")
        assert not hasattr(probe, "layer_weights")

    def test_feature_mode_with_base_model(self) -> None:
        """Test WeightedLSTMProbe in feature mode with base_model."""
        embedding_dims = [256]
        num_classes = 5
        batch_size = 2
        seq_len = 50

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            lstm_hidden_size=64,
            num_layers=2,
            bidirectional=True,
        )

        # Test forward pass
        x = torch.randn(batch_size, seq_len, embedding_dims[0])
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "lstm")
        assert not hasattr(probe, "layer_weights")

    def test_single_tensor_case(self) -> None:
        """Test WeightedLSTMProbe with single tensor embeddings."""
        embedding_dims = [256]
        num_classes = 8
        batch_size = 3

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
            lstm_hidden_size=128,
            num_layers=1,
            bidirectional=False,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "lstm")
        assert not hasattr(probe, "layer_weights")

    def test_list_embeddings_same_dimensions(self) -> None:
        """Test WeightedLSTMProbe with list of embeddings having same dimensions."""
        embedding_dims = [256, 256, 256]  # Same dimensions
        num_classes = 6
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
            base_model=base_model,
            layers=["layer1", "layer2", "layer3"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            lstm_hidden_size=64,
            num_layers=1,
            bidirectional=True,
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)  # Raw audio input
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "lstm")
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (len(embedding_dims),)

        # Test that weights are learnable parameters
        assert probe.layer_weights.requires_grad is True

        # Test debug info
        debug_info = probe.debug_info()
        assert debug_info["probe_type"] == "weighted_lstm"
        assert debug_info["has_layer_weights"] is True
        assert len(debug_info["layer_weights"]) == len(embedding_dims)

    def test_list_embeddings_different_dimensions_raises_error(self) -> None:
        """Test that WeightedLSTMProbe raises error for different embedding
        dimensions."""
        embedding_dims = [256, 512, 128]  # Different dimensions

        base_model = MockBaseModel(embedding_dims)

        with pytest.raises(
            ValueError, match="All embeddings must have the same dimension"
        ):
            WeightedLSTMProbe(
                base_model=base_model,
                layers=["layer1", "layer2", "layer3"],
                num_classes=5,
                device="cpu",
                feature_mode=False,
                aggregation="none",
            )

    def test_feature_mode_without_input_dim_raises_error(self) -> None:
        """Test that WeightedLSTMProbe raises error in feature mode without
        input_dim."""
        with pytest.raises(
            ValueError, match="input_dim must be provided when feature_mode=True"
        ):
            WeightedLSTMProbe(
                base_model=None,
                layers=[],
                num_classes=5,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_positional_encoding(self) -> None:
        """Test WeightedLSTMProbe with positional encoding."""
        input_dim = 128
        num_classes = 4
        batch_size = 2
        seq_len = 30
        max_seq_len = 100

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            use_positional_encoding=True,
            max_sequence_length=max_seq_len,
        )

        assert hasattr(probe, "pos_encoding")
        assert probe.pos_encoding.shape == (1, max_seq_len, input_dim)

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_dropout(self) -> None:
        """Test WeightedLSTMProbe with dropout."""
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        dropout_rate = 0.3

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            dropout_rate=dropout_rate,
        )

        assert hasattr(probe, "dropout")
        assert isinstance(probe.dropout, nn.Dropout)
        assert probe.dropout.p == dropout_rate

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_no_dropout(self) -> None:
        """Test WeightedLSTMProbe without dropout."""
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            dropout_rate=0.0,
        )

        assert probe.dropout is None

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_bidirectional_lstm(self) -> None:
        """Test WeightedLSTMProbe with bidirectional LSTM."""
        input_dim = 128
        num_classes = 5
        batch_size = 3
        seq_len = 40
        lstm_hidden_size = 64

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            lstm_hidden_size=lstm_hidden_size,
            bidirectional=True,
        )

        # Check that classifier input dimension accounts for bidirectional
        expected_classifier_input = lstm_hidden_size * 2  # bidirectional
        assert probe.classifier.in_features == expected_classifier_input

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_multiple_lstm_layers(self) -> None:
        """Test WeightedLSTMProbe with multiple LSTM layers."""
        input_dim = 96
        num_classes = 4
        batch_size = 2
        seq_len = 25
        num_layers = 3
        dropout_rate = 0.2

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        # Check LSTM configuration
        assert probe.lstm.num_layers == num_layers
        assert probe.lstm.dropout == dropout_rate

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_freeze_backbone(self) -> None:
        """Test WeightedLSTMProbe with frozen backbone."""
        embedding_dims = [256]
        num_classes = 3
        batch_size = 2

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
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
        """Test WeightedLSTMProbe with dictionary input."""
        input_dim = 64
        num_classes = 2
        batch_size = 2
        seq_len = 30

        probe = WeightedLSTMProbe(
            base_model=None,
            layers=[],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            input_dim=input_dim,
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

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
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

    def test_sequence_length_handling(self) -> None:
        """Test that different sequence lengths are handled correctly."""
        embedding_dims = [64, 64]  # Same dimensions, different sequence lengths
        num_classes = 2
        batch_size = 1

        # Create base model that returns embeddings with different sequence lengths
        class MockBaseModelVariableSeq(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
            ) -> torch.Tensor | list[torch.Tensor]:
                if aggregation == "none":
                    # Return embeddings with different sequence lengths
                    emb1 = torch.randn(batch_size, 30, 64, device=self.device)
                    emb2 = torch.randn(batch_size, 40, 64, device=self.device)
                    return [emb1, emb2]
                else:
                    return torch.randn(batch_size, 35, 64, device=self.device)

        base_model = MockBaseModelVariableSeq(embedding_dims)
        probe = WeightedLSTMProbe(
            base_model=base_model,
            layers=["layer1", "layer2"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        # Test forward pass
        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        input_dim = 32
        num_classes = 2

        probe = WeightedLSTMProbe(
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
            "lstm_hidden_size",
            "num_layers",
            "bidirectional",
            "dropout_rate",
            "max_sequence_length",
            "use_positional_encoding",
            "target_length",
            "has_layer_weights",
        ]

        for key in expected_keys:
            assert key in debug_info

        assert debug_info["probe_type"] == "weighted_lstm"
        assert debug_info["feature_mode"] is True
        assert debug_info["has_layer_weights"] is False

    def test_cleanup_hooks(self) -> None:
        """Test that hooks are properly cleaned up."""
        embedding_dims = [128]

        base_model = MockBaseModel(embedding_dims)
        probe = WeightedLSTMProbe(
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
        probe = WeightedLSTMProbe(
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

        probe = WeightedLSTMProbe(
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
