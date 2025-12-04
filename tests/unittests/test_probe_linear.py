"""Tests for LinearProbe."""

import pytest
import torch

from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.linear_probe import LinearProbe


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
        """Mock deregister_all_hooks method."""
        self._hooks.clear()


class TestLinearProbe:
    """Test cases for LinearProbe."""

    @pytest.fixture(scope="class")
    def base_model_single(self) -> MockBaseModel:
        """Create a base model with single embedding dimension.

        Returns:
            MockBaseModel: A mock base model with a single embedding dimension of 256.
        """
        return MockBaseModel([256])

    @pytest.fixture(scope="class")
    def base_model_multi_same(self) -> MockBaseModel:
        """Create a base model with multiple same-dimension embeddings.

        Returns:
            MockBaseModel: A mock base model with three embeddings, each of dimension 256.
        """
        return MockBaseModel([256, 256, 256])

    @pytest.fixture(scope="class")
    def base_model_multi_different(self) -> MockBaseModel:
        """Create a base model with multiple different-dimension embeddings.

        Returns:
            MockBaseModel: A mock base model with three embeddings of dimensions 256, 512, and 128.
        """
        return MockBaseModel([256, 512, 128])

    @pytest.fixture(autouse=True)
    def cleanup_hooks(self, request: pytest.FixtureRequest) -> None:
        """Ensure hooks are cleaned up after each test.

        Args:
            request: Pytest request object to access test fixtures.

        Yields:
            None: Yields control to the test, then cleans up hooks after.
        """
        yield
        for fixture_name in ["base_model_single", "base_model_multi_same", "base_model_multi_different"]:
            if fixture_name in request.fixturenames:
                base_model = request.getfixturevalue(fixture_name)
                base_model.deregister_all_hooks()

    def test_feature_mode_configurations(self) -> None:
        """Test feature mode with different configurations."""
        # Feature mode with input_dim
        probe1 = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=10,
            device="cpu",
            feature_mode=True,
            input_dim=512,
        )
        x1 = torch.randn(4, 512)
        output1 = probe1(x1)
        assert output1.shape == (4, 10)
        assert probe1.feature_mode is True
        assert hasattr(probe1, "classifier")
        assert not hasattr(probe1, "layer_weights")

        # Feature mode with base_model
        base_model = MockBaseModel([256])
        probe2 = LinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=5,
            device="cpu",
            feature_mode=True,
        )
        x2 = torch.randn(2, 256)
        output2 = probe2(x2)
        assert output2.shape == (2, 5)
        assert probe2.feature_mode is True

        # Feature mode with dict input
        probe3 = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=2,
            device="cpu",
            feature_mode=True,
            input_dim=64,
        )
        x3 = {
            "raw_wav": torch.randn(2, 64),
            "padding_mask": torch.ones(2, 64, dtype=torch.bool),
        }
        output3 = probe3(x3)
        assert output3.shape == (2, 2)

        # Feature mode error handling
        with pytest.raises(ValueError, match="input_dim must be provided when feature_mode=True"):
            LinearProbe(
                base_model=None,
                layers=[],
                num_classes=5,
                device="cpu",
                feature_mode=True,
                input_dim=None,
            )

    def test_probe_with_base_model(
        self, base_model_single: MockBaseModel, base_model_multi_same: MockBaseModel
    ) -> None:
        """Test probe with base model in different modes."""
        # Single tensor case
        probe1 = LinearProbe(
            base_model=base_model_single,
            layers=["layer1"],
            num_classes=8,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
        )
        x1 = torch.randn(3, 1000)
        output1 = probe1(x1)
        assert output1.shape == (3, 8)
        assert probe1.feature_mode is False
        assert hasattr(probe1, "classifier")

        # Freeze backbone
        probe2 = LinearProbe(
            base_model=base_model_single,
            layers=["layer1"],
            num_classes=3,
            device="cpu",
            feature_mode=False,
            freeze_backbone=True,
        )
        assert probe2.freeze_backbone is True
        x2 = torch.randn(2, 1000)
        output2 = probe2(x2)
        assert output2.shape == (2, 3)

    def test_multi_layer_embeddings(
        self, base_model_multi_same: MockBaseModel, base_model_multi_different: MockBaseModel
    ) -> None:
        """Test probe with multiple layer embeddings."""
        # Same dimensions - creates layer weights
        probe1 = LinearProbe(
            base_model=base_model_multi_same,
            layers=["layer1", "layer2", "layer3"],
            num_classes=6,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )
        x1 = torch.randn(2, 1000)
        output1 = probe1(x1)
        assert output1.shape == (2, 6)
        assert hasattr(probe1, "layer_weights")
        assert probe1.layer_weights.shape == (3,)
        assert probe1.layer_weights.requires_grad is True

        # Test weighted sum behavior
        with torch.no_grad():
            probe1.layer_weights[0] = 1.0
            probe1.layer_weights[1] = 0.0
            probe1.layer_weights[2] = 0.0
        output1_weighted = probe1(x1)
        assert output1_weighted.shape == (2, 6)
        weights = torch.softmax(probe1.layer_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

        # Different dimensions - creates projectors
        probe2 = LinearProbe(
            base_model=base_model_multi_different,
            layers=["layer1", "layer2", "layer3"],
            num_classes=5,
            device="cpu",
            feature_mode=False,
            aggregation="none",
        )
        assert hasattr(probe2, "embedding_projectors")
        assert probe2.embedding_projectors is not None
        assert len(probe2.embedding_projectors) == 3
        x2 = torch.randn(2, 1000)
        output2 = probe2(x2)
        assert output2.shape == (2, 5)

    def test_3d_embedding_handling(self) -> None:
        """Test handling of 3D embeddings by reshaping them."""

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
                    return [torch.randn(batch_size, 10, 128, device=self.device)]
                else:
                    return torch.randn(batch_size, 10, 128, device=self.device)

        base_model = MockBaseModel3D([128])
        probe = LinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=5,
            device="cpu",
            feature_mode=False,
            aggregation="mean",
        )
        x = torch.randn(2, 1000)
        output = probe(x)
        assert output.shape == (2, 5)

    def test_probe_parameters_and_debug_info(self) -> None:
        """Test probe parameters and debug info."""
        # Target length parameter
        probe1 = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=3,
            device="cpu",
            feature_mode=True,
            input_dim=96,
            target_length=2000,
        )
        assert probe1.target_length == 2000
        x1 = torch.randn(2, 96)
        output1 = probe1(x1)
        assert output1.shape == (2, 3)

        # Debug info
        probe2 = LinearProbe(
            base_model=None,
            layers=[],
            num_classes=2,
            device="cpu",
            feature_mode=True,
            input_dim=32,
        )
        debug_info = probe2.debug_info()
        expected_keys = [
            "probe_type",
            "layers",
            "feature_mode",
            "aggregation",
            "freeze_backbone",
            "target_length",
            "has_layer_weights",
        ]
        for key in expected_keys:
            assert key in debug_info
        assert debug_info["probe_type"] == "linear"
        assert debug_info["feature_mode"] is True
        assert debug_info["has_layer_weights"] is False

    def test_hook_cleanup(self) -> None:
        """Test hook cleanup on deletion."""
        embedding_dims = [128]
        base_model = MockBaseModel(embedding_dims)
        probe = LinearProbe(
            base_model=base_model,
            layers=["layer1"],
            num_classes=2,
            device="cpu",
            feature_mode=False,
        )
        assert len(base_model._hooks) == 0
        del probe
        assert len(base_model._hooks) == 0
