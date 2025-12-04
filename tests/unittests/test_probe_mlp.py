"""Tests for MLPProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.mlp_probe import MLPProbe


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
            return [torch.randn(batch_size, dim, device=self.device) for dim in self.embedding_dims]
        return torch.randn(batch_size, self.embedding_dims[0], device=self.device)


class TestMLPProbe:
    """Test cases for MLPProbe."""

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
            MockBaseModel: A mock base model with three embeddings of dimensions 32, 64, and 32.
        """
        return MockBaseModel([32, 64, 32])

    def test_mlp_feature_mode(self) -> None:
        """Test MLP probe in feature mode."""
        probe = MLPProbe(
            base_model=None,
            layers=[],
            num_classes=4,
            device="cpu",
            feature_mode=True,
            input_dim=128,
        )
        x = torch.randn(2, 128)
        out = probe(x)
        assert out.shape == (2, 4)

    def test_mlp_with_base_model(self, base_model_single: MockBaseModel, base_model_multi: MockBaseModel) -> None:
        """Test MLP probe with base model in different configurations."""
        # Single layer
        probe1 = MLPProbe(
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
        probe2 = MLPProbe(
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
