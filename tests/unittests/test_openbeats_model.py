"""Tests for OpenBEATs model embedding extraction functionality.

Tests the unified BEATs model with model_variant='openbeats'.
"""

import pytest
import torch

from representation_learning.models.beats_model import Model
from representation_learning.models.beats.beats import (
    BEATs,
    BEATsConfig,
    BEATS_BASE_CONFIG,
    BEATS_LARGE_CONFIG,
)


class TestBEATsConfig:
    """Test BEATs configuration class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BEATsConfig()

        assert config.embed_dim == 512
        assert config.encoder_layers == 12
        assert config.encoder_embed_dim == 768
        assert config.encoder_attention_heads == 12
        assert config.activation_fn == "gelu"

    def test_config_update(self) -> None:
        """Test configuration update from dictionary."""
        config = BEATsConfig()
        config.update({"encoder_layers": 24, "encoder_embed_dim": 1024})

        assert config.encoder_layers == 24
        assert config.encoder_embed_dim == 1024

    def test_config_from_dict(self) -> None:
        """Test configuration initialization from dictionary."""
        config = BEATsConfig({"encoder_layers": 6})

        assert config.encoder_layers == 6

    def test_config_to_dict(self) -> None:
        """Test configuration conversion to dictionary."""
        config = BEATsConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "encoder_layers" in config_dict
        assert config_dict["encoder_layers"] == 12


class TestBEATsBackbone:
    """Test BEATs backbone model."""

    @pytest.fixture
    def base_config(self) -> BEATsConfig:
        """Create a small base configuration for testing."""
        # Use smaller config for faster testing
        return BEATsConfig({
            "input_patch_size": 16,
            "embed_dim": 64,
            "encoder_layers": 2,
            "encoder_embed_dim": 128,
            "encoder_ffn_embed_dim": 256,
            "encoder_attention_heads": 4,
            "conv_pos": 32,
            "conv_pos_groups": 4,
            "relative_position_embedding": True,
            "num_buckets": 32,
            "max_distance": 128,
            "gru_rel_pos": True,
            "layer_norm_first": True,
        })

    @pytest.fixture
    def beats_backbone(self, base_config: BEATsConfig) -> BEATs:
        """Create a BEATs backbone for testing."""
        return BEATs(base_config)

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor."""
        return torch.randn(2, 16000)  # 1 second at 16kHz

    def test_backbone_initialization(self, beats_backbone: BEATs) -> None:
        """Test backbone initialization."""
        assert beats_backbone is not None
        assert hasattr(beats_backbone, "patch_embedding")
        assert hasattr(beats_backbone, "encoder")
        assert hasattr(beats_backbone, "layer_norm")

    def test_backbone_forward(
        self, beats_backbone: BEATs, sample_audio: torch.Tensor
    ) -> None:
        """Test backbone forward pass."""
        features, padding_mask = beats_backbone(sample_audio)

        assert features.dim() == 3  # (batch, time, features)
        assert features.shape[0] == 2  # batch size
        assert features.shape[2] == 128  # encoder_embed_dim

    def test_backbone_preprocess(
        self, beats_backbone: BEATs, sample_audio: torch.Tensor
    ) -> None:
        """Test audio preprocessing."""
        fbank = beats_backbone.preprocess(sample_audio)

        assert fbank.dim() == 3  # (batch, time, mel_bins)
        assert fbank.shape[0] == 2  # batch size
        assert fbank.shape[2] == 128  # mel bins


class TestOpenBEATsModelWrapper:
    """Test BEATs model wrapper with openbeats variant for the training loop."""

    @pytest.fixture
    def openbeats_model(self) -> Model:
        """Create a BEATs model wrapper with openbeats variant for testing (no pretrained)."""
        return Model(
            num_classes=10,
            pretrained=False,  # Don't load pretrained weights for unit tests
            return_features_only=True,
            device="cpu",
            model_variant="openbeats",
            model_size="base",  # Use smaller model for testing
            disable_layerdrop=True,
        )

    @pytest.fixture
    def openbeats_model_with_classifier(self) -> Model:
        """Create a BEATs model with classifier for testing."""
        return Model(
            num_classes=10,
            pretrained=False,
            return_features_only=False,
            device="cpu",
            model_variant="openbeats",
            model_size="base",
            disable_layerdrop=True,
        )

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor."""
        return torch.randn(2, 16000)  # 1 second at 16kHz

    def test_model_initialization(self, openbeats_model: Model) -> None:
        """Test model wrapper initialization."""
        assert openbeats_model is not None
        assert hasattr(openbeats_model, "backbone")
        assert openbeats_model._return_features_only is True

    def test_model_initialization_with_classifier(
        self, openbeats_model_with_classifier: Model
    ) -> None:
        """Test model wrapper initialization with classifier."""
        assert openbeats_model_with_classifier is not None
        assert hasattr(openbeats_model_with_classifier, "classifier")
        assert openbeats_model_with_classifier.classifier is not None

    def test_model_forward_features_only(
        self, openbeats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test model forward pass with features only."""
        features = openbeats_model(sample_audio)

        assert features.dim() == 2  # (batch, features) - pooled
        assert features.shape[0] == 2  # batch size
        assert features.shape[1] == 768  # encoder_embed_dim for base

    def test_model_forward_with_classifier(
        self, openbeats_model_with_classifier: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test model forward pass with classifier."""
        logits = openbeats_model_with_classifier(sample_audio)

        assert logits.dim() == 2  # (batch, num_classes)
        assert logits.shape[0] == 2  # batch size
        assert logits.shape[1] == 10  # num_classes

    def test_extract_embeddings_with_hooks(
        self, openbeats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test embedding extraction with hooks."""
        # Register hooks for the post_extract_proj layer
        openbeats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = openbeats_model.extract_embeddings(sample_audio, aggregation="mean")

        # Clean up
        openbeats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert embeddings.shape[1] > 0  # features

    def test_extract_embeddings_dict_input(
        self, openbeats_model: Model, sample_audio: torch.Tensor
    ) -> None:
        """Test embedding extraction with dictionary input."""
        input_dict = {"raw_wav": sample_audio}

        # Register hooks
        openbeats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        embeddings = openbeats_model.extract_embeddings(input_dict, aggregation="mean")

        # Clean up
        openbeats_model.deregister_all_hooks()

        assert embeddings.shape[0] == 2  # batch size
        assert torch.is_tensor(embeddings)

    def test_discover_embedding_layers(self, openbeats_model: Model) -> None:
        """Test layer discovery for embeddings."""
        openbeats_model._discover_embedding_layers()

        # Should find fc2 layers in encoder
        assert len(openbeats_model._layer_names) > 0
        # All discovered layers should be fc2 layers
        for layer_name in openbeats_model._layer_names:
            assert ".fc2" in layer_name

    def test_model_validation_error_without_num_classes(self) -> None:
        """Test that model raises error without num_classes when classifier needed."""
        with pytest.raises(ValueError, match="num_classes must be provided"):
            Model(
                num_classes=None,
                pretrained=False,
                return_features_only=False,
                device="cpu",
            )


class TestBEATsModelConfigurations:
    """Test different BEATs model configurations."""

    def test_large_config_values(self) -> None:
        """Test large model configuration values."""
        config = BEATsConfig(BEATS_LARGE_CONFIG)

        assert config.encoder_layers == 24
        assert config.encoder_embed_dim == 1024
        assert config.encoder_attention_heads == 16

    def test_base_config_values(self) -> None:
        """Test base model configuration values."""
        config = BEATsConfig(BEATS_BASE_CONFIG)

        assert config.encoder_layers == 12
        assert config.encoder_embed_dim == 768
        assert config.encoder_attention_heads == 12


class TestOpenBEATsEdgeCases:
    """Test edge cases for BEATs model with openbeats variant."""

    @pytest.fixture
    def openbeats_model(self) -> Model:
        """Create a BEATs model with openbeats variant for testing."""
        return Model(
            num_classes=10,
            pretrained=False,
            return_features_only=True,
            device="cpu",
            model_variant="openbeats",
            model_size="base",
            disable_layerdrop=True,
        )

    def test_empty_audio_raises_error(self, openbeats_model: Model) -> None:
        """Test that empty audio raises an error."""
        empty_audio = torch.tensor([])

        openbeats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        with pytest.raises(ValueError, match="Audio tensor cannot be empty"):
            openbeats_model.extract_embeddings(empty_audio)

        openbeats_model.deregister_all_hooks()

    def test_none_input_raises_error(self, openbeats_model: Model) -> None:
        """Test that None input raises an error."""
        openbeats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        with pytest.raises(ValueError, match="Input tensor cannot be None"):
            openbeats_model.extract_embeddings(None)

        openbeats_model.deregister_all_hooks()

    def test_no_hooks_raises_error(self, openbeats_model: Model) -> None:
        """Test that extracting embeddings without hooks raises error."""
        sample_audio = torch.randn(2, 16000)

        # Ensure no hooks are registered
        openbeats_model.deregister_all_hooks()

        with pytest.raises(ValueError, match="No hooks are registered"):
            openbeats_model.extract_embeddings(sample_audio)

    def test_variable_length_audio(self, openbeats_model: Model) -> None:
        """Test model with different audio lengths."""
        short_audio = torch.randn(2, 8000)  # 0.5 seconds
        long_audio = torch.randn(2, 32000)  # 2 seconds

        openbeats_model.register_hooks_for_layers(["backbone.post_extract_proj"])

        # Both should work
        embeddings_short = openbeats_model.extract_embeddings(short_audio, aggregation="mean")
        embeddings_long = openbeats_model.extract_embeddings(long_audio, aggregation="mean")

        openbeats_model.deregister_all_hooks()

        # Same batch size and feature dimensions
        assert embeddings_short.shape[0] == embeddings_long.shape[0]
        assert embeddings_short.shape[1] == embeddings_long.shape[1]
