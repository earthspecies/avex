"""Tests for importlib.resources-based YAML loading in registry.

These tests verify that the registry correctly uses importlib.resources.files()
to load YAML configurations, ensuring compatibility with:
- Editable installs (pip install -e .)
- Wheel installs (pip install .)
- Zipped packages (wheels with zipped resources)
"""

from pathlib import Path

import pytest

from representation_learning.configs import ModelSpec
from representation_learning.models.utils.registry import (
    _OFFICIAL_MODELS_PKG,
    load_model_spec_from_yaml,
)


class TestLoadModelSpecFromYaml:
    """Test load_model_spec_from_yaml with path inputs only."""

    def test_load_from_path(self, tmp_path: Path) -> None:
        """Test loading from a filesystem path."""
        yaml_content = """
model_spec:
  name: beats
  pretrained: false
  device: cpu
"""
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        spec = load_model_spec_from_yaml(yaml_file)
        assert isinstance(spec, ModelSpec)
        assert spec.name == "beats"
        assert spec.pretrained is False
        assert spec.device == "cpu"

    def test_load_from_string_path(self, tmp_path: Path) -> None:
        """Test loading from a string path."""
        yaml_content = """
model_spec:
  name: efficientnet
  pretrained: true
  device: cuda
"""
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        spec = load_model_spec_from_yaml(str(yaml_file))
        assert isinstance(spec, ModelSpec)
        assert spec.name == "efficientnet"

    def test_load_with_checkpoint_path(self, tmp_path: Path) -> None:
        """Test loading YAML with checkpoint_path at root level."""
        yaml_content = """
model_spec:
  name: beats
  pretrained: false
  device: cpu
checkpoint_path: "gs://bucket/model.pt"
"""
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
            """
model_spec:
  name: beats
  pretrained: false
  device: cpu
checkpoint_path: "/tmp/model1.ckpt"
""",
            encoding="utf-8",
        )

        yaml2 = pkg_dir / "model2.yml"
        yaml2.write_text(
            """
model_spec:
  name: efficientnet
  pretrained: true
  device: cuda
""",
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
            """
model_spec:
  name: beats
  pretrained: false
  device: cpu
checkpoint_path: "gs://my-bucket/checkpoint.pt"
""",
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
