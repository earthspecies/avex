"""Tests for LSTMProbe (duplicated from original LSTM tests)."""

from typing import List, Optional, Union

import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.lstm_probe import (
    LSTMProbe,
)


class MockAudioProcessor:
    def __init__(self, target_length: int = 1000, sr: int = 16000) -> None:
        self.target_length = target_length
        self.sr = sr
        self.target_length_seconds = target_length / sr


class MockBaseModel(ModelBase):
    def __init__(self, embedding_dims: list, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.device = device
        self.embedding_dims = embedding_dims
        self.audio_processor = MockAudioProcessor()

    def extract_embeddings(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.shape[0]
        if aggregation == "none":
            return [
                torch.randn(batch_size, 10, dim, device=self.device) for dim in self.embedding_dims
            ]
        return torch.randn(batch_size, 10, self.embedding_dims[0], device=self.device)


def test_lstm_feature_mode_with_input_dim() -> None:
    input_dim = 64
    num_classes = 3
    probe = LSTMProbe(
        base_model=None,
        layers=[],
        num_classes=num_classes,
        device="cpu",
        feature_mode=True,
        input_dim=input_dim,
    )
    x = torch.randn(2, 10, input_dim)
    out = probe(x)
    assert out.shape == (2, num_classes)


def test_lstm_with_base_model_single() -> None:
    base = MockBaseModel([64])
    probe = LSTMProbe(
        base_model=base,
        layers=["layer1"],
        num_classes=3,
        device="cpu",
        feature_mode=False,
        aggregation="mean",
    )
    x = torch.randn(2, 1000)
    out = probe(x)
    assert out.shape == (2, 3)


def test_lstm_multi_layer_weighted_sum() -> None:
    base = MockBaseModel([32, 64])
    probe = LSTMProbe(
        base_model=base,
        layers=["l1", "l2"],
        num_classes=2,
        device="cpu",
        feature_mode=False,
        aggregation="none",
    )
    x = torch.randn(1, 1000)
    out = probe(x)
    assert out.shape == (1, 2)
