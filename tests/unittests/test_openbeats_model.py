"""Tests for OpenBEATs model embedding extraction functionality."""

import pytest
import torch

from representation_learning.models.openbeats_model import Model
from representation_learning.models.openbeats.openbeats import (
    OpenBEATs,
    OpenBEATsConfig,
    OPENBEATS_BASE_CONFIG,
    OPENBEATS_LARGE_CONFIG,
    OPENBEATS_GIANT_CONFIG,
    OPENBEATS_TITAN_CONFIG,
)


class TestOpenBEATsConfig:
    """Test OpenBEATs configuration class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OpenBEATsConfig()

        assert config.input_patch_size == 16
        assert config.embed_dim == 512
        assert config.encoder_layers == 12
        assert config.encoder_embed_dim == 768
        assert config.encoder_attention_heads == 12
        assert config.activation_fn == "gelu"
        assert config.use_flash_attn is False

    def test_config_update(self) -> None:
        """Test configuration update from dictionary."""
        config = OpenBEATsConfig()
        config.update({"encoder_layers": 24, "encoder_embed_dim": 1024})

        assert config.encoder_layers == 24
        assert config.encoder_embed_dim == 1024

    def test_config_from_dict(self) -> None:
        """Test configuration initialization from dictionary."""
        config = OpenBEATsConfig({"encoder_layers": 6, "use_flash_attn": True})

        assert config.encoder_layers == 6
        assert config.use_flash_attn is True

    def test_config_to_dict(self) -> None:
        """Test configuration conversion to dictionary."""
        config = OpenBEATsConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "encoder_layers" in config_dict
        assert config_dict["encoder_layers"] == 12


class TestOpenBEATsBackbone:
    """Test OpenBEATs backbone model."""

    @pytest.fixture
    def base_config(self) -> OpenBEATsConfig:
        """Create a small base configuration for testing."""
        # Use smaller config for faster testing
        return OpenBEATsConfig({
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
    def openbeats_backbone(self, base_config: OpenBEATsConfig) -> OpenBEATs:
        """Create an OpenBEATs backbone for testing."""
        return OpenBEATs(base_config)

    @pytest.fixture
    def sample_audio(self) -> torch.Tensor:
        """Create sample audio tensor."""
        return torch.randn(2, 16000)  # 1 second at 16kHz

    def test_backbone_initialization(self, openbeats_backbone: OpenBEATs) -> None:
        """Test backbone initialization."""
        assert openbeats_backbone is not None
        assert hasattr(openbeats_backbone, "patch_embedding")
        assert hasattr(openbeats_backbone, "encoder")
        assert hasattr(openbeats_backbone, "layer_norm")

    def test_backbone_forward(
        self, openbeats_backbone: OpenBEATs, sample_audio: torch.Tensor
    ) -> None:
        """Test backbone forward pass."""
        features, padding_mask = openbeats_backbone(sample_audio)

        assert features.dim() == 3  # (batch, time, features)
        assert features.shape[0] == 2  # batch size
        assert features.shape[2] == 128  # encoder_embed_dim

    def test_backbone_preprocess(
        self, openbeats_backbone: OpenBEATs, sample_audio: torch.Tensor
    ) -> None:
        """Test audio preprocessing."""
        fbank = openbeats_backbone.preprocess(sample_audio)

        assert fbank.dim() == 3  # (batch, time, mel_bins)
        assert fbank.shape[0] == 2  # batch size
        assert fbank.shape[2] == 128  # mel bins


class TestOpenBEATsModelWrapper:
    """Test OpenBEATs model wrapper for the training loop."""

    @pytest.fixture
    def openbeats_model(self) -> Model:
        """Create an OpenBEATs model wrapper for testing (no pretrained)."""
        return Model(
            num_classes=10,
            pretrained=False,  # Don't load pretrained weights for unit tests
            return_features_only=True,
            device="cpu",
            model_size="base",  # Use smaller model for testing
            disable_layerdrop=True,
        )

    @pytest.fixture
    def openbeats_model_with_classifier(self) -> Model:
        """Create an OpenBEATs model with classifier for testing."""
        return Model(
            num_classes=10,
            pretrained=False,
            return_features_only=False,
            device="cpu",
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


class TestOpenBEATsModelConfigurations:
    """Test different OpenBEATs model configurations."""

    def test_large_config_values(self) -> None:
        """Test large model configuration values."""
        config = OpenBEATsConfig(OPENBEATS_LARGE_CONFIG)

        assert config.encoder_layers == 24
        assert config.encoder_embed_dim == 1024
        assert config.encoder_attention_heads == 16

    def test_base_config_values(self) -> None:
        """Test base model configuration values."""
        config = OpenBEATsConfig(OPENBEATS_BASE_CONFIG)

        assert config.encoder_layers == 12
        assert config.encoder_embed_dim == 768
        assert config.encoder_attention_heads == 12

    def test_giant_config_values(self) -> None:
        """Test giant model configuration values (~1B parameters).
        
        Giant is a new model size introduced in OpenBEATs, not available in original BEATs.
        """
        config = OpenBEATsConfig(OPENBEATS_GIANT_CONFIG)

        assert config.encoder_layers == 48
        assert config.encoder_embed_dim == 1408
        assert config.encoder_attention_heads == 22
        assert config.encoder_ffn_embed_dim == 6144

    def test_titan_config_values(self) -> None:
        """Test titan model configuration values (~1.9B parameters).
        
        Titan is a new model size introduced in OpenBEATs, not available in original BEATs.
        """
        config = OpenBEATsConfig(OPENBEATS_TITAN_CONFIG)

        assert config.encoder_layers == 64
        assert config.encoder_embed_dim == 1664
        assert config.encoder_attention_heads == 26
        assert config.encoder_ffn_embed_dim == 6656

    def test_flash_attention_config(self) -> None:
        """Test flash attention configuration."""
        config = OpenBEATsConfig({"use_flash_attn": True})

        assert config.use_flash_attn is True


class TestOpenBEATsEdgeCases:
    """Test edge cases for OpenBEATs model."""

    @pytest.fixture
    def openbeats_model(self) -> Model:
        """Create an OpenBEATs model for testing."""
        return Model(
            num_classes=10,
            pretrained=False,
            return_features_only=True,
            device="cpu",
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
