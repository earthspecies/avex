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

from avex import load_model, register_model
from avex.configs import AudioConfig, ModelSpec
from avex.models.base_model import ModelBase
from avex.models.utils.load import (
    _extract_num_classes_from_checkpoint,
    _get_classification_layer_dim_from_state_dict,
    _load_checkpoint,
    _load_from_modelspec,
    load_label_mapping,
)
from avex.models.utils.registry import (
    register_model_class,
)
from avex.utils.utils import _process_state_dict


class TestLoadModel:
    """Test load_model function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from avex.models.utils import registry

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
        model = load_model("test_model", device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert getattr(model, "return_features_only", False) is True
        assert model.num_classes is None

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

        model = load_model(str(yaml_file), device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert getattr(model, "return_features_only", False) is True
        assert model.num_classes is None

    def test_loads_model_from_path_object(self, tmp_path: Path) -> None:
        """Test loading model from Path object."""
        yaml_content = """model_spec:
            name: test_model_type
            pretrained: false
            device: cpu
            """
        yaml_file = tmp_path / "test_model.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        model = load_model(yaml_file, device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert getattr(model, "return_features_only", False) is True
        assert model.num_classes is None

    def test_loads_model_from_modelspec(self) -> None:
        """Test loading model from ModelSpec object."""
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )

        model = load_model(model_spec, device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert getattr(model, "return_features_only", False) is True
        assert model.num_classes is None

    def test_raises_value_error_for_unknown_model(self) -> None:
        """Test that ValueError is raised for unknown model identifier."""
        with pytest.raises(ValueError, match="Unknown model identifier"):
            load_model("nonexistent_model", device="cpu")

    def test_raises_type_error_for_invalid_type(self) -> None:
        """Test that TypeError is raised for invalid model type."""
        with pytest.raises(TypeError, match="Unsupported model type"):
            load_model(123, device="cpu")  # type: ignore[arg-type]


class TestCreateModel:
    """Legacy create_model tests (kept for historical context)."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from avex.models.utils import registry

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


class TestGetClassificationLayerDim:
    """Test _get_classification_layer_dim_from_state_dict function."""

    def test_extracts_from_classifier_weight(self) -> None:
        """Test extracting num_classes from classifier.weight."""
        state_dict = {
            "classifier.weight": torch.randn(25, 128),
            "classifier.bias": torch.randn(25),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes == 25

    def test_extracts_from_classifier_bias(self) -> None:
        """Test extracting num_classes from classifier.bias when weight not present."""
        state_dict = {
            "classifier.bias": torch.randn(30),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes == 30

    def test_extracts_from_head_weight(self) -> None:
        """Test extracting num_classes from head.weight."""
        state_dict = {
            "head.weight": torch.randn(42, 256),
            "head.bias": torch.randn(42),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes == 42

    def test_extracts_from_classification_head(self) -> None:
        """Test extracting num_classes from classification_head."""
        state_dict = {
            "classification_head.weight": torch.randn(100, 512),
            "classification_head.bias": torch.randn(100),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes == 100

    def test_prefers_last_classifier_when_multiple(self) -> None:
        """Test that the function prefers the last classifier when multiple exist."""
        state_dict = {
            "classifier.weight": torch.randn(10, 128),
            "classifier.bias": torch.randn(10),
            "head.weight": torch.randn(20, 256),
            "head.bias": torch.randn(20),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should prefer "head" over "classifier" (alphabetically last)
        assert num_classes == 20

    def test_handles_prefixed_keys(self) -> None:
        """Test extracting from state dict with module. or model. prefixes."""
        state_dict = {
            "model.classifier.weight": torch.randn(15, 64),
            "model.classifier.bias": torch.randn(15),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes == 15

    def test_excludes_backbone_classifier(self) -> None:
        """Test that backbone.classifier is excluded."""
        state_dict = {
            "backbone.classifier.weight": torch.randn(50, 128),
            "classifier.weight": torch.randn(25, 128),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should use classifier, not backbone.classifier
        assert num_classes == 25

    def test_excludes_encoder_head(self) -> None:
        """Test that encoder.head is excluded."""
        state_dict = {
            "encoder.head.weight": torch.randn(50, 128),
            "head.weight": torch.randn(30, 128),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should use head, not encoder.head
        assert num_classes == 30

    def test_excludes_fc_layers(self) -> None:
        """Test that fc1, fc2, fc3 layers are excluded."""
        state_dict = {
            "fc1.weight": torch.randn(100, 128),
            "fc2.weight": torch.randn(50, 100),
            "classifier.weight": torch.randn(25, 50),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should use classifier, not fc1 or fc2
        assert num_classes == 25

    def test_returns_none_when_no_classifier(self) -> None:
        """Test that None is returned when no classifier keys exist."""
        state_dict = {
            "backbone.layer1.weight": torch.randn(64, 3, 3, 3),
            "backbone.layer2.weight": torch.randn(128, 64, 3, 3),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes is None

    def test_returns_none_when_classifier_wrong_shape(self) -> None:
        """Test that None is returned when classifier has wrong shape."""
        state_dict = {
            "classifier.weight": torch.randn(10, 64, 3),  # 3D tensor (wrong shape)
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes is None

    def test_returns_none_when_bias_wrong_shape(self) -> None:
        """Test that None is returned when bias has wrong shape."""
        state_dict = {
            "classifier.bias": torch.randn(10, 5),  # 2D tensor (wrong shape for bias)
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes is None

    def test_returns_dimension_even_when_large(self) -> None:
        """Test that dimension is returned even when large (no size limit)."""
        state_dict = {
            "classifier.weight": torch.randn(50000, 128),  # Large dimension
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should return the dimension regardless of size
        assert num_classes == 50000

    def test_handles_empty_state_dict(self) -> None:
        """Test that None is returned for empty state dict."""
        state_dict = {}
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        assert num_classes is None

    def test_prefers_weight_over_bias(self) -> None:
        """Test that weight is preferred over bias when both exist."""
        state_dict = {
            "classifier.weight": torch.randn(40, 128),
            "classifier.bias": torch.randn(40),
        }
        processed_state_dict = _process_state_dict(state_dict, keep_classifier=True)

        num_classes = _get_classification_layer_dim_from_state_dict(processed_state_dict)

        # Should use weight (checked first in reversed sorted order)
        assert num_classes == 40


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


class TestLoadLabelMapping:
    """Test load_label_mapping function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        """Set up model registry for testing.

        Yields:
            None: Fixture yields nothing, just sets up the registry.
        """
        from avex.models.utils import registry

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

        mapping = load_label_mapping(str(json_file))

        assert mapping is not None
        assert mapping["label_to_index"]["class_a"] == 0
        assert mapping["index_to_label"][0] == "class_a"
        assert len(mapping["label_to_index"]) == 3

    def test_loads_from_path_object(self, tmp_path: Path) -> None:
        """Test loading class mapping from Path object."""
        mapping_data = {"label1": 0, "label2": 1}
        json_file = tmp_path / "mapping.json"
        json_file.write_text(json.dumps(mapping_data), encoding="utf-8")

        mapping = load_label_mapping(json_file)

        assert mapping is not None
        assert mapping["label_to_index"]["label1"] == 0

    def test_returns_none_when_file_not_found(self, tmp_path: Path) -> None:
        """Test that None is returned when file doesn't exist."""
        json_file = tmp_path / "nonexistent.json"

        mapping = load_label_mapping(str(json_file))

        assert mapping is None

    def test_returns_none_when_invalid_json(self, tmp_path: Path) -> None:
        """Test that None is returned when JSON is invalid."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json", encoding="utf-8")

        mapping = load_label_mapping(str(json_file))

        assert mapping is None

    def test_returns_none_when_not_dict(self, tmp_path: Path) -> None:
        """Test that None is returned when JSON is not a dictionary."""
        json_file = tmp_path / "not_dict.json"
        json_file.write_text("[1, 2, 3]", encoding="utf-8")

        mapping = load_label_mapping(str(json_file))

        assert mapping is None


class TestLoadCheckpoint:
    """Test _load_checkpoint function."""

    @pytest.fixture
    def test_model(self) -> ModelBase:
        """Create a test model for checkpoint loading.

        Yields:
            ModelBase: A test model instance.
        """
        from avex.models.utils import registry

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
        from avex.models.utils import registry

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

    def test_loads_with_pretrained_true(self) -> None:
        """Test loading model with pretrained=True (no checkpoint)."""
        from avex.models.utils import registry

        # Register model spec with pretrained=True
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=True,  # Has embedded pretrained weights
            device="cpu",
        )
        register_model("test_pretrained_model", model_spec)

        model = load_model("test_pretrained_model", device="cpu", return_features_only=True)

        assert isinstance(model, ModelBase)
        assert getattr(model, "return_features_only", False) is True

        registry._MODEL_REGISTRY.clear()

    def test_raises_error_when_num_classes_required(self) -> None:
        """Test that ValueError is raised when classifier creation is requested without checkpoint."""
        from avex.models.utils import registry

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

        with pytest.raises(
            ValueError,
            match="no longer creates new classifier heads",
        ):
            _load_from_modelspec(
                model_spec,
                device="cpu",
                checkpoint_path=None,
                registry_key="no_features_test",
                return_features_only=False,
            )

        registry._MODEL_CLASSES.clear()
        registry._MODEL_REGISTRY.clear()
