"""Tests for model factory functions.

This module tests the factory functions for building model instances from
registered classes and ModelSpec configurations.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.base_model import ModelBase
from representation_learning.models.utils.factory import (
    _add_model_spec_params,
    build_model,
    build_model_from_spec,
)
from representation_learning.models.utils.registry import (
    register_model,
    register_model_class,
)


class TestAddModelSpecParams:
    """Test _add_model_spec_params internal function."""

    def test_adds_params_when_not_none(self) -> None:
        """Test that params are added when they exist and are not None."""
        init_kwargs: dict[str, Any] = {}
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
            text_model_name="test_model",
            efficientnet_variant="b1",
            use_naturelm=True,
        )

        _add_model_spec_params(init_kwargs, model_spec)

        assert init_kwargs["text_model_name"] == "test_model"
        assert init_kwargs["efficientnet_variant"] == "b1"
        assert init_kwargs["use_naturelm"] is True

    def test_skips_params_when_none(self) -> None:
        """Test that params are skipped when they are None."""
        init_kwargs: dict[str, Any] = {}
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
            text_model_name=None,
            efficientnet_variant="b1",
        )

        _add_model_spec_params(init_kwargs, model_spec)

        assert "text_model_name" not in init_kwargs
        assert init_kwargs["efficientnet_variant"] == "b1"

    def test_skips_params_when_empty_string(self) -> None:
        """Test that params are skipped when they are empty string."""
        init_kwargs: dict[str, Any] = {}
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
            text_model_name="",
            efficientnet_variant="b1",
        )

        _add_model_spec_params(init_kwargs, model_spec)

        assert "text_model_name" not in init_kwargs
        assert init_kwargs["efficientnet_variant"] == "b1"

    def test_skips_params_when_not_exist(self) -> None:
        """Test that params are skipped when they don't exist on model_spec.

        Note: ModelSpec has default values for some fields, so we test with
        a simple object that doesn't have the attributes at all.
        """
        init_kwargs: dict[str, Any] = {}
        # Create a simple object without the optional params

        class SimpleSpec:
            """Simple spec without optional params."""

            def __init__(self) -> None:
                self.name = "test_model"
                # No text_model_name, efficientnet_variant, etc.

        model_spec = SimpleSpec()

        _add_model_spec_params(init_kwargs, model_spec)

        # Should not add any params since SimpleSpec doesn't have those attributes
        assert len(init_kwargs) == 0


class TestBuildModel:
    """Test build_model function."""

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
                **kwargs: object,
            ) -> None:
                # Convert dict to AudioConfig if needed (like build_model_from_spec does)
                if isinstance(audio_config, dict):
                    audio_config = AudioConfig(**audio_config)
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.custom_param = kwargs.get("custom_param", None)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes or 10)

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

    def test_builds_model_successfully(self) -> None:
        """Test successful model building (backbone-only)."""
        model = build_model("test_model", device="cpu")

        assert isinstance(model, ModelBase)
        assert model.device == "cpu"

    def test_builds_model_with_additional_kwargs(self) -> None:
        """Test building model with additional kwargs."""
        model = build_model("test_model", device="cpu", custom_param="test_value")

        assert isinstance(model, ModelBase)
        assert model.custom_param == "test_value"

    def test_raises_value_error_when_model_spec_not_found(self) -> None:
        """Test that ValueError is raised when ModelSpec is not found."""
        with pytest.raises(ValueError, match="No ModelSpec found for 'nonexistent_model'"):
            build_model("nonexistent_model", device="cpu")

    def test_raises_key_error_when_model_class_not_registered(self) -> None:
        """Test that KeyError is raised when model class is not registered."""
        from representation_learning.models.utils import registry

        # Register a model spec with unregistered class
        model_spec = ModelSpec(name="unregistered_type", pretrained=False, device="cpu")
        register_model("test_unregistered", model_spec)

        with pytest.raises(KeyError, match="Model class 'unregistered_type' is not registered"):
            build_model("test_unregistered", device="cpu")

        # Clean up
        registry._MODEL_REGISTRY.clear()

    def test_raises_exception_when_model_instantiation_fails(self) -> None:
        """Test that exceptions during model instantiation are propagated."""

        # Create a model class that raises an exception
        @register_model_class
        class FailingModelClass(ModelBase):
            """Model class that fails during initialization."""

            name = "failing_model_type"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device, audio_config=None)
                raise RuntimeError("Initialization failed")

        # Register model spec
        model_spec = ModelSpec(name="failing_model_type", pretrained=False, device="cpu")
        register_model("failing_model", model_spec)

        with pytest.raises(RuntimeError, match="Initialization failed"):
            build_model("failing_model", device="cpu")

        # Clean up
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()


class TestBuildModelFromSpec:
    """Test build_model_from_spec function."""

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
                audio_config: AudioConfig | None = None,
                **kwargs: object,
            ) -> None:
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.audio_config = audio_config  # Store for testing
                self.custom_param = kwargs.get("custom_param", None)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes or 10)

        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_CLASSES.clear()

    def test_builds_model_from_spec_with_audio_config_object(self) -> None:
        """Test building model from spec with AudioConfig object."""
        audio_config = AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=10)
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
            audio_config=audio_config,
        )

        model = build_model_from_spec(model_spec, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.audio_config is not None
        assert model.audio_config.sample_rate == 16000

    def test_builds_model_from_spec_with_audio_config_dict(self) -> None:
        """Test building model from spec with audio_config as dict."""
        audio_config_dict = {
            "sample_rate": 22050,
            "representation": "mel_spectrogram",
            "target_length_seconds": 5,
        }
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
            audio_config=audio_config_dict,  # type: ignore[arg-type]
        )

        model = build_model_from_spec(model_spec, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.audio_config is not None
        assert model.audio_config.sample_rate == 22050

    def test_builds_model_from_spec_without_audio_config(self) -> None:
        """Test building model from spec without audio_config."""
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
            audio_config=None,
        )

        model = build_model_from_spec(model_spec, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.audio_config is None

    def test_builds_model_from_spec_without_num_classes(self) -> None:
        """Test building model from spec without num_classes."""
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )

        model = build_model_from_spec(model_spec, device="cpu")

        assert isinstance(model, ModelBase)

    def test_builds_model_from_spec_with_additional_kwargs(self) -> None:
        """Test building model from spec with additional kwargs.

        Note: build_model_from_spec filters kwargs based on signature.
        Since TestModelClass accepts **kwargs, custom_param should pass through.
        """
        model_spec = ModelSpec(
            name="test_model_type",
            pretrained=False,
            device="cpu",
        )

        # custom_param is passed via **kwargs, which should be preserved
        # because TestModelClass.__init__ has **kwargs in its signature
        model = build_model_from_spec(model_spec, device="cpu", custom_param="test_value")

        assert isinstance(model, ModelBase)

    def test_filters_invalid_params(self) -> None:
        """Test that invalid params are filtered out before model instantiation."""

        # Create a model class that doesn't accept certain params
        @register_model_class
        class RestrictedModelClass(ModelBase):
            """Model class with restricted parameters."""

            name = "restricted_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int | None = None,
                audio_config: AudioConfig | None = None,
                # Note: no custom_param or invalid_param
            ) -> None:
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes or 10)

        model_spec = ModelSpec(
            name="restricted_model_type",
            pretrained=False,
            device="cpu",
            efficientnet_variant="b1",  # This param won't be in the model's __init__
        )

        # Should not raise an error even though efficientnet_variant is not accepted
        model = build_model_from_spec(model_spec, device="cpu", invalid_param="should_be_filtered")

        assert isinstance(model, ModelBase)

        # Clean up
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()

    def test_adds_model_spec_params(self) -> None:
        """Test that model-specific params from ModelSpec are added."""

        @register_model_class
        class ParamModelClass(ModelBase):
            """Model class that accepts model-specific params."""

            name = "param_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int | None = None,
                audio_config: AudioConfig | None = None,
                efficientnet_variant: str | None = None,
                use_naturelm: bool | None = None,
            ) -> None:
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.efficientnet_variant = efficientnet_variant
                self.use_naturelm = use_naturelm
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor.
                """
                return torch.zeros(x.shape[0], self.num_classes or 10)

        model_spec = ModelSpec(
            name="param_model_type",
            pretrained=False,
            device="cpu",
            efficientnet_variant="b1",  # Use valid variant
            use_naturelm=True,
        )

        model = build_model_from_spec(model_spec, device="cpu")

        assert isinstance(model, ModelBase)
        assert model.efficientnet_variant == "b1"
        assert model.use_naturelm is True

        # Clean up
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()

    def test_raises_key_error_when_model_class_not_registered(self) -> None:
        """Test that KeyError is raised when model class is not registered."""
        model_spec = ModelSpec(name="unregistered_type", pretrained=False, device="cpu")

        with pytest.raises(KeyError, match="Model class 'unregistered_type' is not registered"):
            build_model_from_spec(model_spec, device="cpu")

    def test_raises_exception_when_model_instantiation_fails(self) -> None:
        """Test that exceptions during model instantiation are propagated."""

        # Create a model class that raises an exception
        @register_model_class
        class FailingModelClass(ModelBase):
            """Model class that fails during initialization."""

            name = "failing_model_type"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device, audio_config=None)
                raise ValueError("Initialization failed")

        model_spec = ModelSpec(name="failing_model_type", pretrained=False, device="cpu")

        with pytest.raises(ValueError, match="Initialization failed"):
            build_model_from_spec(model_spec, device="cpu")

        # Clean up
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()
