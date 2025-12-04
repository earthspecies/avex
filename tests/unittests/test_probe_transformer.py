"""Tests for TransformerProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.transformer_probe import TransformerProbe


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
        padding_mask: torch.Tensor | None = None,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        batch_size = x.shape[0]
        if aggregation == "none":
            return [torch.randn(batch_size, 20, dim, device=self.device) for dim in self.embedding_dims]
        return torch.randn(batch_size, 20, self.embedding_dims[0], device=self.device)


class TestTransformerProbe:
    """Test cases for TransformerProbe."""

    @pytest.fixture(scope="class")
    def base_model_single(self) -> MockBaseModel:
        """Create a base model with single embedding dimension.

        Returns:
            MockBaseModel: A mock base model with a single embedding dimension of 64.
        """
        return MockBaseModel([64])

    @pytest.fixture(scope="class")
    def base_model_multi(self) -> MockBaseModel:
        """Create a base model with multiple embeddings.

        Returns:
            MockBaseModel: A mock base model with three embeddings of dimensions 32, 64, and 96.
        """
        return MockBaseModel([32, 64, 96])

    def test_transformer_feature_mode(self) -> None:
        """Test transformer probe in feature mode."""
        probe = TransformerProbe(
            base_model=None,
            layers=[],
            num_classes=4,
            device="cpu",
            feature_mode=True,
            input_dim=96,
        )
        x = torch.randn(2, 10, 96)
        out = probe(x)
        assert out.shape == (2, 4)

    def test_transformer_with_base_model(
        self, base_model_single: MockBaseModel, base_model_multi: MockBaseModel
    ) -> None:
        """Test transformer probe with base model in different configurations."""
        # Single layer
        probe1 = TransformerProbe(
            base_model=base_model_single,
            layers=["layer1"],
            num_classes=3,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
        )
        x1 = torch.randn(2, 1000)
        out1 = probe1(x1)
        assert out1.shape == (2, 3)

        # Multi-layer weighted sum
        probe2 = TransformerProbe(
            base_model=base_model_multi,
            layers=["l1", "l2", "l3"],
            num_classes=2,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )
        x2 = torch.randn(2, 1000)
        out2 = probe2(x2)
        assert out2.shape == (2, 2)
