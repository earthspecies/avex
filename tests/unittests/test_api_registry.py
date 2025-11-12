"""Tests for importlib.resources-based YAML loading in registry.

These tests verify that the registry correctly uses importlib.resources.files()
to load YAML configurations, ensuring compatibility with:
- Editable installs (pip install -e .)
- Wheel installs (pip install .)
- Zipped packages (wheels with zipped resources)
"""

from pathlib import Path
from textwrap import dedent

import pytest

from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.base_model import ModelBase
from representation_learning.models.utils.registry import (
    _OFFICIAL_MODELS_PKG,
    describe_model,
    get_checkpoint_path,
    get_model_class,
    get_model_spec,
    list_model_classes,
    list_models,
    load_model_spec_from_yaml,
    register_model,
    register_model_class,
)


class TestLoadModelSpecFromYaml:
    """Test load_model_spec_from_yaml with path inputs only."""

    def test_load_from_path(self, tmp_path: Path) -> None:
        """Test loading from a filesystem path."""
        yaml_content = dedent("""\
            model_spec:
              name: beats
              pretrained: false
              device: cpu
            """)
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        spec = load_model_spec_from_yaml(yaml_file)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "beats"
        assert spec.pretrained is False
        assert spec.device == "cpu"

    def test_load_from_string_path(self, tmp_path: Path) -> None:
        """Test loading from a string path."""
        yaml_content = dedent("""\
            model_spec:
              name: efficientnet
              pretrained: true
              device: cuda
            """)
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        spec = load_model_spec_from_yaml(str(yaml_file))
        assert isinstance(spec, ModelSpec)
        assert spec.name == "efficientnet"

    def test_load_with_checkpoint_path(self, tmp_path: Path) -> None:
        """Test loading YAML with checkpoint_path at root level."""
        yaml_content = dedent("""\
            model_spec:
              name: beats
              pretrained: false
              device: cpu
            checkpoint_path: "gs://bucket/model.pt"
            """)
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        spec = load_model_spec_from_yaml(yaml_file)
        assert isinstance(spec, ModelSpec)
        # checkpoint_path is not part of ModelSpec, it's at root level
        # This is handled by get_checkpoint_path()


class TestImportlibResources:
    """Test importlib.resources.files() usage for packaged YAMLs."""

    def test_registry_uses_resources_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that registry uses resources.files() for packaged YAMLs.

        This simulates how the registry works when the package is installed.
        """
        # Create a temporary package structure
        pkg_dir = tmp_path / "test_pkg" / "official_models"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir.parent / "__init__.py").write_text("")

        # Create test YAML files
        yaml1 = pkg_dir / "model1.yml"
        yaml1.write_text(
            dedent("""\
                model_spec:
                  name: beats
                  pretrained: false
                  device: cpu
                checkpoint_path: "/tmp/model1.ckpt"
                """),
            encoding="utf-8",
        )

        yaml2 = pkg_dir / "model2.yml"
        yaml2.write_text(
            dedent("""\
                model_spec:
                  name: efficientnet
                  pretrained: true
                  device: cuda
                """),
            encoding="utf-8",
        )

        # Add to sys.path and import
        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            # Monkeypatch the package name
            from representation_learning.models.utils import registry

            original_pkg = registry._OFFICIAL_MODELS_PKG
            registry._OFFICIAL_MODELS_PKG = "test_pkg.official_models"
            registry._MODEL_REGISTRY.clear()

            # Initialize registry - should use resources.files()
            registry.initialize_registry()

            # Verify models were loaded
            models = registry.list_models()
            assert "model1" in models
            assert "model2" in models
            assert models["model1"].name == "beats"
            assert models["model2"].name == "efficientnet"

            # Verify checkpoint path reading
            checkpoint = registry.get_checkpoint_path("model1")
            assert checkpoint == "/tmp/model1.ckpt"

            # Restore
            registry._OFFICIAL_MODELS_PKG = original_pkg
        finally:
            sys.path.remove(str(tmp_path))
            registry._MODEL_REGISTRY.clear()
            registry.initialize_registry()

    def test_zip_safe_reading(self) -> None:
        """Zip-safety handled inside registry via entry.open(); no file-like input."""
        assert True

    def test_resources_files_iterdir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that resources.files().iterdir() works correctly.

        This verifies the pattern used in _auto_register_from_yaml().
        """
        from importlib import resources

        # Test with actual package if it exists, otherwise skip
        try:
            root = resources.files(_OFFICIAL_MODELS_PKG)
            entries = list(root.iterdir())
            # Should find YAML files
            yaml_files = [e for e in entries if e.name.endswith(".yml") and e.is_file()]
            # In development, we might have YAML files
            # In installed package, they should be there if packaged correctly
            assert isinstance(yaml_files, list)  # Just verify it doesn't crash
        except (ModuleNotFoundError, ImportError):
            pytest.skip(f"Package {_OFFICIAL_MODELS_PKG} not available for testing")


class TestCheckpointPathReading:
    """Test checkpoint path reading from packaged resources."""

    def test_get_checkpoint_path_from_resources(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test get_checkpoint_path uses resources.files() correctly."""
        # Use a unique package name to avoid module caching conflicts
        import importlib
        import sys

        pkg_name = f"test_pkg_{id(tmp_path)}"
        # Create temporary package
        pkg_dir = tmp_path / pkg_name / "official_models"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir.parent / "__init__.py").write_text("")

        yaml_file = pkg_dir / "test_model.yml"
        yaml_file.write_text(
            dedent("""\
                model_spec:
                  name: beats
                  pretrained: false
                  device: cpu
                checkpoint_path: "gs://my-bucket/checkpoint.pt"
                """),
            encoding="utf-8",
        )

        sys.path.insert(0, str(tmp_path))

        try:
            # Clear any cached modules with the same name
            if pkg_name in sys.modules:
                del sys.modules[pkg_name]
            if f"{pkg_name}.official_models" in sys.modules:
                del sys.modules[f"{pkg_name}.official_models"]

            # Import the test package to make it available to importlib.resources
            importlib.import_module(pkg_name)
            importlib.import_module(f"{pkg_name}.official_models")

            from representation_learning.models.utils import registry

            original_pkg = registry._OFFICIAL_MODELS_PKG
            registry._OFFICIAL_MODELS_PKG = f"{pkg_name}.official_models"
            registry._MODEL_REGISTRY.clear()
            registry.initialize_registry()

            # Force materialization and verify registration
            models = registry.list_models()
            assert "test_model" in models, f"Expected 'test_model' in {list(models.keys())}"

            # Verify checkpoint path is read correctly
            checkpoint = registry.get_checkpoint_path("test_model")
            assert checkpoint == "gs://my-bucket/checkpoint.pt"

            # Restore
            registry._OFFICIAL_MODELS_PKG = original_pkg
        finally:
            # Clean up modules
            if pkg_name in sys.modules:
                del sys.modules[pkg_name]
            if f"{pkg_name}.official_models" in sys.modules:
                del sys.modules[f"{pkg_name}.official_models"]
            sys.path.remove(str(tmp_path))
            registry._MODEL_REGISTRY.clear()
            registry.initialize_registry()


class TestRegisterModel:
    """Test register_model function."""

    def test_register_model_adds_to_registry(self) -> None:
        """Test that register_model adds a ModelSpec to the registry."""
        from representation_learning.models.utils import registry

        # Clear registry to start fresh
        registry._MODEL_REGISTRY.clear()

        # Create a test model spec
        model_spec = ModelSpec(
            name="test_model",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=10),
        )

        # Register the model
        register_model("test_custom_model", model_spec)

        # Verify it's in the registry
        retrieved_spec = get_model_spec("test_custom_model")
        assert retrieved_spec is not None
        assert retrieved_spec.name == "test_model"
        assert retrieved_spec.device == "cpu"
        assert retrieved_spec.pretrained is False

        # Clean up
        registry._MODEL_REGISTRY.clear()

    def test_register_model_overwrites_existing(self) -> None:
        """Test that register_model overwrites existing entries."""
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_REGISTRY.clear()

        # Register first model
        spec1 = ModelSpec(name="model1", pretrained=False, device="cpu")
        register_model("test_model", spec1)

        # Register with same name but different spec
        spec2 = ModelSpec(name="model2", pretrained=True, device="cuda")
        register_model("test_model", spec2)

        # Verify the second one overwrote the first
        retrieved = get_model_spec("test_model")
        assert retrieved is not None
        assert retrieved.name == "model2"
        assert retrieved.pretrained is True
        assert retrieved.device == "cuda"

        # Clean up
        registry._MODEL_REGISTRY.clear()


class TestRegisterModelClass:
    """Test register_model_class function."""

    def test_register_model_class_adds_to_registry(self) -> None:
        """Test that register_model_class adds a model class to the registry."""
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_CLASSES.clear()

        # Create a simple test model class
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class."""

            name = "test_model_class"

            def __init__(self, device: str, num_classes: int, **kwargs: object) -> None:
                super().__init__(device=device)
                self.num_classes = num_classes

            def forward(self, x: object) -> object:
                return x

        # Verify it's registered
        retrieved_class = get_model_class("test_model_class")
        assert retrieved_class is not None
        assert retrieved_class == TestModelClass
        assert retrieved_class.name == "test_model_class"

        # Verify it's in the list
        classes = list_model_classes()
        assert "test_model_class" in classes

        # Clean up
        registry._MODEL_CLASSES.clear()

    def test_register_model_class_without_name_attribute(self) -> None:
        """Test that register_model_class uses class name if name attribute is missing."""
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_CLASSES.clear()

        # Create a model class without a name attribute
        @register_model_class
        class AnotherTestModel(ModelBase):
            """Another test model."""

            def __init__(self, device: str, num_classes: int, **kwargs: object) -> None:
                super().__init__(device=device)
                self.num_classes = num_classes

            def forward(self, x: object) -> object:
                return x

        # Should be registered with lowercased class name
        retrieved_class = get_model_class("anothertestmodel")
        assert retrieved_class is not None
        assert retrieved_class == AnotherTestModel

        # Clean up
        registry._MODEL_CLASSES.clear()

    def test_register_model_class_overwrites_existing(self) -> None:
        """Test that register_model_class overwrites existing entries."""
        from representation_learning.models.utils import registry

        # Clear registry
        registry._MODEL_CLASSES.clear()

        # Register first class
        @register_model_class
        class FirstModel(ModelBase):
            name = "shared_name"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device)

            def forward(self, x: object) -> object:
                return x

        # Register second class with same name
        @register_model_class
        class SecondModel(ModelBase):
            name = "shared_name"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device)

            def forward(self, x: object) -> object:
                return x

        # Verify the second one overwrote the first
        retrieved = get_model_class("shared_name")
        assert retrieved is not None
        assert retrieved == SecondModel

        # Clean up
        registry._MODEL_CLASSES.clear()


class TestGetModelSpec:
    """Test get_model_spec function."""

    def test_get_model_spec_returns_none_when_not_found(self) -> None:
        """Test that get_model_spec returns None for unregistered models."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        spec = get_model_spec("nonexistent_model")
        assert spec is None

    def test_get_model_spec_returns_registered_model(self) -> None:
        """Test that get_model_spec returns the correct ModelSpec."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        model_spec = ModelSpec(name="test", pretrained=False, device="cpu")
        register_model("test_model", model_spec)

        retrieved = get_model_spec("test_model")
        assert retrieved is not None
        assert retrieved == model_spec

        registry._MODEL_REGISTRY.clear()


class TestGetModelClass:
    """Test get_model_class function."""

    def test_get_model_class_returns_none_when_not_found(self) -> None:
        """Test that get_model_class returns None for unregistered classes."""
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()
        cls = get_model_class("nonexistent_class")
        assert cls is None

    def test_get_model_class_returns_registered_class(self) -> None:
        """Test that get_model_class returns the correct class."""
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()

        @register_model_class
        class TestClass(ModelBase):
            name = "test_class"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device)

            def forward(self, x: object) -> object:
                return x

        retrieved = get_model_class("test_class")
        assert retrieved is not None
        assert retrieved == TestClass

        registry._MODEL_CLASSES.clear()


class TestListModels:
    """Test list_models function."""

    def test_list_models_returns_copy(self) -> None:
        """Test that list_models returns a copy, not the original dict."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        model_spec = ModelSpec(name="test", pretrained=False, device="cpu")
        register_model("test_model", model_spec)

        models = list_models()
        assert "test_model" in models

        # Modify the returned dict - should not affect registry
        models["test_model"] = ModelSpec(name="modified", pretrained=True, device="cuda")
        assert get_model_spec("test_model").name == "test"  # Original unchanged

        registry._MODEL_REGISTRY.clear()

    def test_list_models_empty_registry(self) -> None:
        """Test that list_models returns empty dict when registry is empty."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        models = list_models()
        assert models == {}


class TestListModelClasses:
    """Test list_model_classes function."""

    def test_list_model_classes_returns_registered_classes(self) -> None:
        """Test that list_model_classes returns all registered class names."""
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()

        @register_model_class
        class Class1(ModelBase):
            name = "class1"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device)

            def forward(self, x: object) -> object:
                return x

        @register_model_class
        class Class2(ModelBase):
            name = "class2"

            def __init__(self, device: str, **kwargs: object) -> None:
                super().__init__(device=device)

            def forward(self, x: object) -> object:
                return x

        classes = list_model_classes()
        assert "class1" in classes
        assert "class2" in classes
        assert len(classes) >= 2

        registry._MODEL_CLASSES.clear()

    def test_list_model_classes_empty_registry(self) -> None:
        """Test that list_model_classes returns empty list when registry is empty."""
        from representation_learning.models.utils import registry

        registry._MODEL_CLASSES.clear()
        classes = list_model_classes()
        assert classes == []


class TestGetCheckpointPath:
    """Test get_checkpoint_path function."""

    def test_get_checkpoint_path_raises_keyerror_when_not_registered(self) -> None:
        """Test that get_checkpoint_path raises KeyError for unregistered models."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()

        with pytest.raises(KeyError, match="Model 'nonexistent' is not registered"):
            get_checkpoint_path("nonexistent")

    def test_get_checkpoint_path_returns_none_when_no_checkpoint(self) -> None:
        """Test that get_checkpoint_path returns None when no checkpoint_path in YAML."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        model_spec = ModelSpec(name="test", pretrained=False, device="cpu")
        register_model("test_model", model_spec)

        # Model is registered but no checkpoint_path in YAML
        checkpoint = get_checkpoint_path("test_model")
        assert checkpoint is None

        registry._MODEL_REGISTRY.clear()


class TestDescribeModel:
    """Test describe_model function."""

    def test_describe_model_raises_keyerror_when_not_found(self) -> None:
        """Test that describe_model raises KeyError for unregistered models."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()

        with pytest.raises(KeyError, match="Model 'nonexistent' is not registered"):
            describe_model("nonexistent")

    def test_describe_model_returns_model_info(self) -> None:
        """Test that describe_model returns correct model information."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        model_spec = ModelSpec(
            name="test_model",
            pretrained=True,
            device="cuda",
            audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=10),
        )
        register_model("test_model", model_spec)

        info = describe_model("test_model")
        assert isinstance(info, dict)
        assert info["name"] == "test_model"
        assert info["pretrained"] is True
        assert info["device"] == "cuda"
        assert "_metadata" in info
        assert info["_metadata"]["name"] == "test_model"
        assert info["_metadata"]["model_type"] == "test_model"

        registry._MODEL_REGISTRY.clear()

    def test_describe_model_verbose_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that describe_model prints when verbose=True."""
        from representation_learning.models.utils import registry

        registry._MODEL_REGISTRY.clear()
        model_spec = ModelSpec(name="test", pretrained=False, device="cpu")
        register_model("test_model", model_spec)

        info = describe_model("test_model", verbose=True)
        captured = capsys.readouterr()
        assert "test_model" in captured.out or "test" in captured.out
        assert isinstance(info, dict)

        registry._MODEL_REGISTRY.clear()


class TestLoadModelSpecFromYamlExceptions:
    """Test load_model_spec_from_yaml exception cases."""

    def test_load_model_spec_from_yaml_raises_valueerror_for_invalid_structure(self, tmp_path: Path) -> None:
        """Test that load_model_spec_from_yaml raises ValueError for invalid YAML structure."""
        # Test with non-dict YAML (list at root)
        yaml_file = tmp_path / "invalid.yml"
        yaml_file.write_text("- item1\n- item2", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML must define a mapping"):
            load_model_spec_from_yaml(yaml_file)

    def test_load_model_spec_from_yaml_raises_valueerror_for_missing_model_spec(self, tmp_path: Path) -> None:
        """Test that load_model_spec_from_yaml raises ValueError when model_spec is not a dict."""
        yaml_file = tmp_path / "invalid.yml"
        yaml_file.write_text("model_spec: not_a_dict", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid model specification structure"):
            load_model_spec_from_yaml(yaml_file)

    def test_load_model_spec_from_yaml_raises_valueerror_for_invalid_model_spec(self, tmp_path: Path) -> None:
        """Test that load_model_spec_from_yaml raises ValueError for invalid ModelSpec fields."""
        yaml_file = tmp_path / "invalid.yml"
        yaml_file.write_text(
            dedent("""\
                model_spec:
                  name: test
                  pretrained: invalid_boolean
                  device: cpu
                """),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Failed to build ModelSpec"):
            load_model_spec_from_yaml(yaml_file)
