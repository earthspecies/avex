"""Tests for the new embedding aggregation functionality."""

from typing import List, Optional, Union

import pytest
import torch

from representation_learning.configs import ProbeConfig
from representation_learning.models.probes import get_probe


class TestEmbeddingAggregation:
    """Test the new embedding aggregation functionality."""

    def test_linear_probe_with_different_aggregations(self) -> None:
        """Test that linear probe works with different aggregation methods."""
        # Test mean aggregation
        probe_config_mean = ProbeConfig(
            probe_type="linear",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe_mean = get_probe(
            probe_config=probe_config_mean,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test max aggregation
        probe_config_max = ProbeConfig(
            probe_type="linear",
            aggregation="max",
            input_processing="pooled",
            target_layers=["layer_12"],
        )

        probe_max = get_probe(
            probe_config=probe_config_max,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        # Test concat aggregation
        probe_config_concat = ProbeConfig(
            probe_type="linear",
            aggregation="none",
            input_processing="pooled",
            target_layers=["layer_8", "layer_12"],
        )

        probe_concat = get_probe(
            probe_config=probe_config_concat,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=1024,  # 2 layers * 512 dim
        )

        assert probe_mean is not None
        assert probe_max is not None
        assert probe_concat is not None

        # Test forward pass with mean aggregation
        x = torch.randn(4, 512)
        output_mean = probe_mean(x)
        assert output_mean.shape == (4, 10)

        # Test forward pass with max aggregation
        output_max = probe_max(x)
        assert output_max.shape == (4, 10)

        # Test forward pass with concat aggregation
        x_concat = torch.randn(4, 1024)
        output_concat = probe_concat(x_concat)
        assert output_concat.shape == (4, 10)

    def test_linear_probe_with_none_aggregation(self) -> None:
        """Test that linear probe works with aggregation=None using projection heads."""

        # Create a mock base model that returns list of embeddings
        class MockBaseModel:
            def __init__(self) -> None:
                self.audio_processor = type(
                    "MockProcessor", (), {"target_length": 24000, "sr": 16000}
                )()

            def register_hooks_for_layers(self, layers: list) -> None:
                # Mock method - no actual implementation needed for testing
                pass

            def extract_embeddings(
                self,
                x: torch.Tensor,
                aggregation: str = "mean",
                padding_mask: Optional[torch.Tensor] = None,
                freeze_backbone: bool = False,
            ) -> Union[torch.Tensor, List[torch.Tensor]]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return list of embeddings for different layers
                    return [
                        torch.randn(batch_size, 512),  # layer_8: 512 dim
                        torch.randn(batch_size, 768),  # layer_12: 768 dim
                    ]
                else:
                    # Return aggregated tensor for other aggregation methods
                    return torch.randn(batch_size, 512)

        mock_model = MockBaseModel()

        # Create the probe directly to test our new functionality
        from representation_learning.models.probes.linear_probe import LinearProbe

        probe_none = LinearProbe(
            base_model=mock_model,
            layers=["layer_8", "layer_12"],
            num_classes=10,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )

        assert probe_none is not None

        # Test that projection heads were created (new name embedding_projectors)
        assert hasattr(probe_none, "embedding_projectors")
        assert len(probe_none.embedding_projectors) == 2

        # Test forward pass with mock embeddings
        x = torch.randn(4, 24000)  # batch_size=4, audio_length=24000
        output_none = probe_none(x)
        assert output_none.shape == (4, 10)

    def test_mlp_probe_with_none_aggregation(self) -> None:
        """Test that MLP probe works with aggregation=None using projection heads."""

        # Create a mock base model that returns list of embeddings
        class MockBaseModel:
            def __init__(self) -> None:
                self.audio_processor = type(
                    "MockProcessor", (), {"target_length": 24000, "sr": 16000}
                )()

            def register_hooks_for_layers(self, layers: list) -> None:
                # Mock method - no actual implementation needed for testing
                pass

            def extract_embeddings(
                self,
                x: torch.Tensor,
                aggregation: str = "mean",
                padding_mask: Optional[torch.Tensor] = None,
                freeze_backbone: bool = False,
            ) -> Union[torch.Tensor, List[torch.Tensor]]:
                batch_size = x.shape[0]
                if aggregation == "none":
                    # Return list of embeddings for different layers
                    return [
                        torch.randn(batch_size, 512),  # layer_8: 512 dim
                        torch.randn(batch_size, 768),  # layer_12: 768 dim
                    ]
                else:
                    # Return aggregated tensor for other aggregation methods
                    return torch.randn(batch_size, 512)

        mock_model = MockBaseModel()

        # Create the probe directly to test our new functionality
        from representation_learning.models.probes.mlp_probe import MLPProbe

        probe_none = MLPProbe(
            base_model=mock_model,
            layers=["layer_8", "layer_12"],
            num_classes=10,
            device="cpu",
            feature_mode=False,
            aggregation="none",
            hidden_dims=[256, 128],
        )

        assert probe_none is not None

        # Test that projection heads were created (new name embedding_projectors)
        assert hasattr(probe_none, "embedding_projectors")
        assert len(probe_none.embedding_projectors) == 2

        # Test forward pass with mock embeddings
        x = torch.randn(4, 24000)  # batch_size=4, audio_length=24000
        output_none = probe_none(x)
        assert output_none.shape == (4, 10)

    def test_mlp_probe_with_different_aggregations(self) -> None:
        """Test that MLP probe works with different aggregation methods."""
        # Test mean aggregation
        probe_config_mean = ProbeConfig(
            probe_type="mlp",
            aggregation="mean",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.1,
            activation="relu",
        )

        # Test max aggregation
        probe_config_max = ProbeConfig(
            probe_type="mlp",
            aggregation="max",
            input_processing="pooled",
            target_layers=["layer_12"],
            hidden_dims=[256, 128],
            dropout_rate=0.1,
            activation="relu",
        )

        probe_mean = get_probe(
            probe_config=probe_config_mean,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        probe_max = get_probe(
            probe_config=probe_config_max,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        assert probe_mean is not None
        assert probe_max is not None

        # Test forward pass
        x = torch.randn(4, 512)
        output_mean = probe_mean(x)
        output_max = probe_max(x)

        assert output_mean.shape == (4, 10)
        assert output_max.shape == (4, 10)

    def test_sequence_probes_with_none_aggregation(self) -> None:
        """Test that sequence probes work with 'none' aggregation."""
        # Test LSTM probe with none aggregation
        probe_config_lstm = ProbeConfig(
            probe_type="lstm",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            lstm_hidden_size=128,
            num_layers=1,
            bidirectional=False,
        )

        # Test attention probe with none aggregation
        probe_config_attention = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=1,
        )

        probe_lstm = get_probe(
            probe_config=probe_config_lstm,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )

        probe_attention = get_probe(
            probe_config=probe_config_attention,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        assert probe_lstm is not None
        assert probe_attention is not None

        # Test forward pass with sequence input
        # Sequence-based probes expect (batch_size, seq_len, embedding_dim)
        x_lstm = torch.randn(4, 10, 512)  # batch_size=4, seq_len=10, features=512
        x_attention = torch.randn(4, 10, 256)  # batch_size=4, seq_len=10, features=256

        output_lstm = probe_lstm(x_lstm)
        output_attention = probe_attention(x_attention)

        assert output_lstm.shape == (4, 10)
        assert output_attention.shape == (4, 10)

    def test_weighted_probes_with_sequence_processing(self) -> None:
        """Test that weighted probes work with sequence input processing."""
        # For sequence processing, use attention and transformer
        probe_config_attention = ProbeConfig(
            probe_type="attention",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=1,
        )

        probe_config_transformer = ProbeConfig(
            probe_type="transformer",
            aggregation="none",
            input_processing="sequence",
            target_layers=["layer_12"],
            num_heads=4,
            attention_dim=256,
            num_layers=1,
        )

        probe_attention = get_probe(
            probe_config=probe_config_attention,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        probe_transformer = get_probe(
            probe_config=probe_config_transformer,
            base_model=None,
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=256,
        )

        assert probe_attention is not None
        assert probe_transformer is not None

        x_sequence = torch.randn(4, 10, 256)
        out_attn = probe_attention(x_sequence)
        out_tx = probe_transformer(x_sequence)

        assert out_attn.shape == (4, 10)
        assert out_tx.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__])
