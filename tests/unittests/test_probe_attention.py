"""Tests for AttentionProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.attention_probe import (
    AttentionProbe,
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
        batch_size = x.shape[0]
        if aggregation == "none":
            embeddings = []
            for dim in self.embedding_dims:
                seq_len = 50
                emb = torch.randn(batch_size, seq_len, dim, device=self.device)
                embeddings.append(emb)
            return embeddings
        else:
            seq_len = 50
            emb_dim = self.embedding_dims[0] if self.embedding_dims else 256
            return torch.randn(batch_size, seq_len, emb_dim, device=self.device)

    def register_hooks_for_layers(self, layers: list) -> None:
        self._hooks = {layer: None for layer in layers}

    def deregister_all_hooks(self) -> None:
        self._hooks.clear()


class TestAttentionProbe:
    """Test cases for AttentionProbe."""

    def test_feature_mode_with_input_dim(self) -> None:
        input_dim = 512
        num_classes = 10
        batch_size = 4
        seq_len = 50
        num_heads = 8
        attention_dim = 512
        num_layers = 2

        probe = AttentionProbe(
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

        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "attention_layers")
        assert len(probe.attention_layers) == num_layers
        assert not hasattr(probe, "layer_weights")

    def test_feature_mode_with_base_model(self) -> None:
        embedding_dims = [256]
        num_classes = 5
        batch_size = 2
        seq_len = 50
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=num_classes,
            device="cpu",
            feature_mode=True,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
        )

        x = torch.randn(batch_size, seq_len, embedding_dims[0])
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is True
        assert hasattr(probe, "attention_layers")
        assert not hasattr(probe, "layer_weights")

    def test_single_tensor_case(self) -> None:
        embedding_dims = [256]
        num_classes = 8
        batch_size = 3
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
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

        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "attention_layers")
        assert not hasattr(probe, "layer_weights")

    def test_list_embeddings_same_dimensions(self) -> None:
        embedding_dims = [256, 256, 256]
        num_classes = 6
        batch_size = 2
        num_heads = 4
        attention_dim = 256
        num_layers = 2

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
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

        x = torch.randn(batch_size, 1000)
        output = probe(x)

        assert output.shape == (batch_size, num_classes)
        assert probe.feature_mode is False
        assert hasattr(probe, "attention_layers")
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (len(embedding_dims),)
        assert probe.layer_weights.requires_grad is True

        debug_info = probe.debug_info()
        assert debug_info["probe_type"] == "attention"
        assert debug_info["has_layer_weights"] is True
        assert len(debug_info["layer_weights"]) == len(embedding_dims)

    def test_list_embeddings_different_dimensions_automatic_projection(self) -> None:
        embedding_dims = [256, 512, 128]
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
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

        x = torch.randn(2, 1000)
        output = probe(x)
        assert output.shape == (2, 5)
        assert hasattr(probe, "embedding_projectors")
        assert probe.embedding_projectors is not None
        assert hasattr(probe, "layer_weights")
        assert probe.layer_weights.shape == (3,)

    def test_feature_mode_without_input_dim_raises_error(self) -> None:
        with pytest.raises(
            ValueError, match="input_dim must be provided when feature_mode=True"
        ):
            AttentionProbe(
                base_model=None,
                layers=[],
                num_classes=5,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_positional_encoding(self) -> None:
        input_dim = 128
        num_classes = 4
        batch_size = 2
        seq_len = 30
        max_seq_len = 100
        num_heads = 4
        attention_dim = 128
        num_layers = 1

        probe = AttentionProbe(
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
        assert probe.pos_encoding.shape == (1, max_seq_len, input_dim)

        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_no_positional_encoding(self) -> None:
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = AttentionProbe(
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
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_dropout_rate(self) -> None:
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1
        dropout_rate = 0.3

        probe = AttentionProbe(
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
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_no_dropout(self) -> None:
        input_dim = 64
        num_classes = 3
        batch_size = 2
        seq_len = 20
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = AttentionProbe(
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
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_multiple_attention_layers(self) -> None:
        input_dim = 96
        num_classes = 4
        batch_size = 2
        seq_len = 25
        num_heads = 6
        attention_dim = 96
        num_layers = 3
        dropout_rate = 0.2

        probe = AttentionProbe(
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

        assert len(probe.attention_layers) == num_layers
        x = torch.randn(batch_size, seq_len, input_dim)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_freeze_backbone(self) -> None:
        embedding_dims = [256]
        num_classes = 3
        batch_size = 2
        num_heads = 4
        attention_dim = 256
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
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
        x = torch.randn(batch_size, 1000)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_dict_input(self) -> None:
        input_dim = 64
        num_classes = 2
        batch_size = 2
        seq_len = 30
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        probe = AttentionProbe(
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

        x = {
            "raw_wav": torch.randn(batch_size, seq_len, input_dim),
            "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

    def test_weighted_sum_behavior(self) -> None:
        embedding_dims = [128, 128, 128]
        num_classes = 3
        batch_size = 2
        num_heads = 4
        attention_dim = 128
        num_layers = 1

        base_model = MockBaseModel(embedding_dims)
        probe = AttentionProbe(
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

        with torch.no_grad():
            probe.layer_weights[0] = 1.0
            probe.layer_weights[1] = 0.0
            probe.layer_weights[2] = 0.0

        x = torch.randn(batch_size, 1000)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)

        weights = torch.softmax(probe.layer_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_sequence_length_handling(self) -> None:
        embedding_dims = [64, 64]
        num_classes = 2
        batch_size = 1
        num_heads = 2
        attention_dim = 64
        num_layers = 1

        class MockBaseModelVariableSeq(MockBaseModel):
            def extract_embeddings(
                self,
                x: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "mean",
                freeze_backbone: bool = True,
            ) -> torch.Tensor | list[torch.Tensor]:
                if aggregation == "none":
                    emb1 = torch.randn(batch_size, 30, 64, device=self.device)
                    emb2 = torch.randn(batch_size, 40, 64, device=self.device)
                    return [emb1, emb2]
                else:
                    return torch.randn(batch_size, 35, 64, device=self.device)

        base_model = MockBaseModelVariableSeq(embedding_dims)
        probe = AttentionProbe(
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

        x = torch.randn(batch_size, 1000)
        output = probe(x)
        assert output.shape == (batch_size, num_classes)
