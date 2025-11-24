"""Tests for model loading functions.

This module tests the load_model and related functions for loading models
with checkpoints, class mappings, and various configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from representation_learning import load_model, register_model
from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.base_model import ModelBase
from representation_learning.models.utils.load import (
    _extract_num_classes_from_checkpoint,
    _load_checkpoint,
    create_model,
    load_class_mapping,
)
from representation_learning.models.utils.registry import (
    register_model_class,
)


class TestLoadModel:
    """Test load_model function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

        # Register a test model class
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class."""

            name = "test_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int | None = None,
                audio_config: dict[str, Any] | AudioConfig | None = None,
                return_features_only: bool = False,
                **kwargs: object,
            ) -> None:
                # Convert dict to AudioConfig if needed
                if isinstance(audio_config, dict):
                    audio_config = AudioConfig(**audio_config)
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.return_features_only = return_features_only
                if not return_features_only and num_classes is not None:
                    self.classifier = torch.nn.Linear(128, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor (embeddings or logits).
                """
                if self.return_features_only:
                    return torch.zeros(x.shape[0], 128)  # Embeddings
                if self.num_classes is None:
                    return torch.zeros(x.shape[0], 10)  # Default
                return torch.zeros(x.shape[0], self.num_classes)

        # Register a model spec
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=10),
        )
        register_model("test_model", model_spec)

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

    def test_loads_registered_model(self) -> None:
        """Test loading a registered model."""
        model = load_model("test_model", num_classes=5, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 5

    def test_loads_model_from_yaml_path(self, tmp_path: Path) -> None:
        """Test loading model from YAML file."""
        yaml_content = """model_spec:
            name: test_model_type
            pretrained: false
            device: cpu
            audio_config:
                sample_rate: 16000
                representation: raw
                target_length_seconds: 10
            """
        yaml_file = tmp_path / "test_model.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        model = load_model(str(yaml_file), num_classes=10, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 10

    def test_loads_model_from_path_object(self, tmp_path: Path) -> None:
        """Test loading model from Path object."""
        yaml_content = """model_spec:
            name: test_model_type
            pretrained: false
            device: cpu
            """
        yaml_file = tmp_path / "test_model.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        model = load_model(yaml_file, num_classes=5, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 5

    def test_loads_model_from_modelspec(self) -> None:
        """Test loading model from ModelSpec object."""
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )

        model = load_model(model_spec, num_classes=8, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 8

    def test_raises_value_error_for_unknown_model(self) -> None:
        """Test that ValueError is raised for unknown model identifier."""
        with pytest.raises(ValueError, match="Unknown model identifier"):
            load_model("nonexistent_model", num_classes=10, device="cpu")

    def test_raises_type_error_for_invalid_type(self) -> None:
        """Test that TypeError is raised for invalid model type."""
        with pytest.raises(TypeError, match="Unsupported model type"):
            load_model(123, num_classes=10, device="cpu")  # type: ignore[arg-type]


class TestCreateModel:
    """Test create_model function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

        # Register a test model class
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class."""

            name = "test_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int,
                audio_config: AudioConfig | None = None,
                **kwargs: object,
            ) -> None:
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.classifier = torch.nn.Linear(128, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes)

        # Register a model spec
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )
        register_model("test_model", model_spec)

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

    def test_creates_model_from_registered_class(self) -> None:
        """Test creating model from registered class (plugin architecture)."""
        model = create_model("test_model_type", num_classes=5, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 5

    def test_creates_model_from_registered_spec(self) -> None:
        """Test creating model from registered spec."""
        model = create_model("test_model", num_classes=10, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 10

    def test_creates_model_from_yaml_path(self, tmp_path: Path) -> None:
        """Test creating model from YAML file."""
        yaml_content = """model_spec:
            name: test_model_type
            pretrained: false
            device: cpu
            """
        yaml_file = tmp_path / "test_model.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        model = create_model(str(yaml_file), num_classes=15, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.num_classes == 15

    def test_raises_value_error_for_unknown_model(self) -> None:
        """Test that ValueError is raised for unknown model identifier."""
        with pytest.raises(ValueError, match="Unknown model identifier"):
            create_model("nonexistent_model", num_classes=10, device="cpu")


class TestExtractNumClassesFromCheckpoint:
    """Test _extract_num_classes_from_checkpoint function."""

    def test_extracts_from_classifier_weight(self, tmp_path: Path) -> None:
        """Test extracting num_classes from classifier.weight."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "classifier.weight": torch.randn(25, 128),
            "classifier.bias": torch.randn(25),
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        assert num_classes == 25

    def test_extracts_from_classifier_bias(self, tmp_path: Path) -> None:
        """Test extracting num_classes from classifier.bias."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "classifier.bias": torch.randn(30),
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        assert num_classes == 30

    def test_extracts_from_metadata_when_classifier_wrong_shape(self, tmp_path: Path) -> None:
        """Test extracting num_classes from metadata when classifier has wrong shape.

        The function checks metadata after processing classifier keys, but only
        if classifier keys exist but don't have the right shape (not 2D for weight,
        not 1D for bias).
        """
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "classification_head.weight": torch.randn(10, 64, 3),  # 3D tensor (wrong shape)
            "num_classes": 42,  # Metadata fallback
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        # Function finds classifier key but wrong shape, so checks metadata
        assert num_classes == 42

    def test_extracts_from_model_config_when_classifier_wrong_shape(self, tmp_path: Path) -> None:
        """Test extracting num_classes from model_config when classifier has wrong shape."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "head.bias": torch.randn(10, 5),  # 2D tensor (wrong shape for bias)
            "model_config": {"num_classes": 50},
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        # Function finds classifier key but wrong shape, so checks model_config
        assert num_classes == 50

    def test_handles_prefixed_keys(self, tmp_path: Path) -> None:
        """Test extracting from checkpoint with module. or model. prefixes."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "model.classifier.weight": torch.randn(20, 128),
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        assert num_classes == 20

    def test_returns_none_when_checkpoint_not_found(self, tmp_path: Path) -> None:
        """Test that None is returned when checkpoint doesn't exist."""
        checkpoint_path = tmp_path / "nonexistent.pt"

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        assert num_classes is None

    def test_returns_none_when_no_classifier(self, tmp_path: Path) -> None:
        """Test that None is returned when checkpoint has no classifier."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "backbone.layer1.weight": torch.randn(64, 3, 3, 3),
        }
        torch.save(checkpoint, checkpoint_path)

        num_classes = _extract_num_classes_from_checkpoint(str(checkpoint_path), "cpu")

        assert num_classes is None


class TestLoadClassMapping:
    """Test load_class_mapping function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()

        # Register a model spec
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
        )
        register_model("test_model_with_mapping", model_spec)

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()

    def test_loads_from_direct_path(self, tmp_path: Path) -> None:
        """Test loading class mapping from direct path."""
        mapping_data = {
            "class_a": 0,
            "class_b": 1,
            "class_c": 2,
        }
        json_file = tmp_path / "mapping.json"
        json_file.write_text(json.dumps(mapping_data), encoding="utf-8")

        mapping = load_class_mapping(str(json_file))

        assert mapping is not None
        assert mapping["label_to_index"]["class_a"] == 0
        assert mapping["index_to_label"][0] == "class_a"
        assert len(mapping["label_to_index"]) == 3

    def test_loads_from_path_object(self, tmp_path: Path) -> None:
        """Test loading class mapping from Path object."""
        mapping_data = {"label1": 0, "label2": 1}
        json_file = tmp_path / "mapping.json"
        json_file.write_text(json.dumps(mapping_data), encoding="utf-8")

        mapping = load_class_mapping(json_file)

        assert mapping is not None
        assert mapping["label_to_index"]["label1"] == 0

    def test_returns_none_when_file_not_found(self, tmp_path: Path) -> None:
        """Test that None is returned when file doesn't exist."""
        json_file = tmp_path / "nonexistent.json"

        mapping = load_class_mapping(str(json_file))

        assert mapping is None

    def test_returns_none_when_invalid_json(self, tmp_path: Path) -> None:
        """Test that None is returned when JSON is invalid."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json", encoding="utf-8")

        mapping = load_class_mapping(str(json_file))

        assert mapping is None

    def test_returns_none_when_not_dict(self, tmp_path: Path) -> None:
        """Test that None is returned when JSON is not a dictionary."""
        json_file = tmp_path / "not_dict.json"
        json_file.write_text("[1, 2, 3]", encoding="utf-8")

        mapping = load_class_mapping(str(json_file))

        assert mapping is None


class TestLoadCheckpoint:
    """Test _load_checkpoint function."""

    @pytest.fixture
    def test_model(self) -> ModelBase:
        """Create a test model for checkpoint loading.

        Yields:
            ModelBase: A test model instance.
        """
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()

        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class."""

            name = "test_model_type"

            def __init__(self, device: str, num_classes: int = 10) -> None:
                super().__init__(device=device, audio_config=None)
                self.num_classes = num_classes
                self.backbone = torch.nn.Linear(128, 64)
                self.classifier = torch.nn.Linear(64, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes)

        model = TestModelClass(device="cpu", num_classes=10)
        yield model

        registry._MODEL_CLASSES.clear()

    def test_loads_checkpoint_with_classifier(self, test_model: ModelBase, tmp_path: Path) -> None:
        """Test loading checkpoint with classifier weights."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        # Save original classifier weights
        original_classifier_weight = test_model.classifier.weight.clone()

        # Create checkpoint with different classifier weights
        checkpoint = {
            "backbone.weight": torch.randn(64, 128),
            "backbone.bias": torch.randn(64),
            "classifier.weight": torch.randn(10, 64),
            "classifier.bias": torch.randn(10),
        }
        torch.save(checkpoint, checkpoint_path)

        _load_checkpoint(test_model, str(checkpoint_path), "cpu", keep_classifier=True)

        # Classifier weights should be updated
        assert not torch.allclose(test_model.classifier.weight, original_classifier_weight)

    def test_loads_checkpoint_without_classifier(self, test_model: ModelBase, tmp_path: Path) -> None:
        """Test loading checkpoint without classifier weights."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        # Save original classifier weights
        original_classifier_weight = test_model.classifier.weight.clone()

        # Create checkpoint with different classifier weights
        checkpoint = {
            "backbone.weight": torch.randn(64, 128),
            "backbone.bias": torch.randn(64),
            "classifier.weight": torch.randn(10, 64),
            "classifier.bias": torch.randn(10),
        }
        torch.save(checkpoint, checkpoint_path)

        _load_checkpoint(test_model, str(checkpoint_path), "cpu", keep_classifier=False)

        # Classifier weights should remain unchanged (not loaded)
        assert torch.allclose(test_model.classifier.weight, original_classifier_weight)

    def test_raises_file_not_found_error(self, test_model: ModelBase, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when checkpoint doesn't exist."""
        checkpoint_path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            _load_checkpoint(test_model, str(checkpoint_path), "cpu", keep_classifier=False)


class TestLoadFromModelSpec:
    """Test _load_from_modelspec internal function behavior."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

        # Register a test model class that supports return_features_only
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class with return_features_only support."""

            name = "test_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int | None = None,
                audio_config: AudioConfig | None = None,
                return_features_only: bool = False,
                **kwargs: object,
            ) -> None:
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.return_features_only = return_features_only
                if not return_features_only and num_classes is not None:
                    self.classifier = torch.nn.Linear(128, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor (embeddings or logits).
                """
                if self.return_features_only:
                    return torch.zeros(x.shape[0], 128)
                if self.num_classes is None:
                    return torch.zeros(x.shape[0], 10)
                return torch.zeros(x.shape[0], self.num_classes)

        # Register a model spec
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )
        register_model("test_model", model_spec)

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

    def test_loads_with_return_features_only(self) -> None:
        """Test loading model with return_features_only=True."""
        model = load_model("test_model", device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert model.return_features_only is True
        assert model.num_classes is None

    def test_extracts_num_classes_from_checkpoint(self, tmp_path: Path) -> None:
        """Test extracting num_classes from checkpoint when num_classes=None."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "classifier.weight": torch.randn(15, 128),
            "classifier.bias": torch.randn(15),
        }
        torch.save(checkpoint, checkpoint_path)

        model = load_model(
            "test_model",
            num_classes=None,
            device="cpu",
            checkpoint_path=str(checkpoint_path),
        )

        assert isinstance(model, ModelBase)
        assert model.num_classes == 15
        assert hasattr(model, "classifier")

    def test_loads_with_explicit_num_classes(self, tmp_path: Path) -> None:
        """Test loading with explicit num_classes (should not load classifier from checkpoint)."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {
            "classifier.weight": torch.randn(20, 128),
            "classifier.bias": torch.randn(20),
        }
        torch.save(checkpoint, checkpoint_path)

        model = load_model(
            "test_model",
            num_classes=10,  # Different from checkpoint
            device="cpu",
            checkpoint_path=str(checkpoint_path),
        )

        assert isinstance(model, ModelBase)
        assert model.num_classes == 10  # Should use explicit value, not checkpoint
        assert model.classifier.weight.shape[0] == 10  # New classifier

    def test_loads_with_pretrained_true(self) -> None:
        """Test loading model with pretrained=True (should use return_features_only)."""
        from representation_learning.models.utils import registry

        # Register model spec with pretrained=True
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=True,  # Has embedded pretrained weights
            device="cpu",
        )
        register_model("test_pretrained_model", model_spec)

        model = load_model("test_pretrained_model", device="cpu")

        assert isinstance(model, ModelBase)
        assert model.return_features_only is True

        registry._MODEL_REGISTRY.clear()

    def test_raises_error_when_num_classes_required(self) -> None:
        """Test that ValueError is raised when num_classes is required but not provided."""
        from representation_learning.models.utils import registry

        # Register a model class that doesn't support return_features_only
        @register_model_class
        class NoFeaturesModel(ModelBase):
            """Model without return_features_only support."""

            name = "no_features_model"

            def __init__(self, device: str, num_classes: int) -> None:  # num_classes required
                super().__init__(device=device, audio_config=None)
                self.num_classes = num_classes
                self.classifier = torch.nn.Linear(128, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes)

        model_spec = ModelSpec(
            name="no_features_model",
            pretrained=False,
            device="cpu",
        )
        register_model("no_features_test", model_spec)

        with pytest.raises(ValueError, match="num_classes must be provided"):
            load_model("no_features_test", num_classes=None, device="cpu")

        registry._MODEL_CLASSES.clear()
        registry._MODEL_REGISTRY.clear()
