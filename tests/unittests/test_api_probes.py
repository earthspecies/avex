"""Tests for probe API functions.

This module tests the probe factory functions for building probe instances
from ProbeConfig objects and listing available model layers for probing.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from avex import list_model_layers, load_model
from avex.configs import AudioConfig, ModelSpec, ProbeConfig
from avex.models.base_model import ModelBase
from avex.models.probes.utils import build_probe_from_config
from avex.models.utils.registry import (
    register_model,
    register_model_class,
)


class TestBuildProbeFromConfig:
    """Test build_probe_from_config function."""

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

        # Register a test model class with layer discovery
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class with discoverable layers."""

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

                # Create some layers for discovery
                self.layer1 = torch.nn.Linear(128, 64)
                self.layer2 = torch.nn.Linear(64, 32)
                if not return_features_only and num_classes is not None:
                    self.classifier = torch.nn.Linear(32, num_classes)
                self.to(device)

            def extract_embeddings(
                self,
                x: torch.Tensor | dict[str, torch.Tensor],
                *,
                padding_mask: torch.Tensor | None = None,
                aggregation: str = "none",
                freeze_backbone: bool = True,  # Accept but ignore (for compatibility)
            ) -> torch.Tensor | list[torch.Tensor]:
                """Extract embeddings using hooks.

                Returns:
                    torch.Tensor | list[torch.Tensor]: Extracted embeddings, either a
                        single tensor or a list of tensors depending on aggregation.

                Raises:
                    ValueError: If no hooks are registered or no layers are found.
                """
                # Clear previous hook outputs
                self._clear_hook_outputs()

                # Ensure hooks are registered
                self.ensure_hooks_registered()

                if not self._hooks:
                    raise ValueError("No hooks registered. Call register_hooks_for_layers() first.")

                # Extract raw audio if provided as dict
                if isinstance(x, dict):
                    wav = x["raw_wav"]
                    mask = x.get("padding_mask")
                else:
                    wav = x
                    mask = padding_mask

                # Forward pass to trigger hooks (use no_grad if freeze_backbone)
                if freeze_backbone:
                    with torch.no_grad():
                        self.forward(wav, mask)
                else:
                    self.forward(wav, mask)

                # Collect embeddings from hook outputs
                embeddings = []
                for layer_name in self._hook_outputs.keys():
                    emb = self._hook_outputs[layer_name]
                    if freeze_backbone:
                        emb = emb.detach()
                    embeddings.append(emb)

                if not embeddings:
                    raise ValueError(f"No layers found matching: {self._hook_outputs.keys()}")

                # Handle aggregation
                if aggregation == "none":
                    return embeddings if len(embeddings) > 1 else embeddings[0]
                elif aggregation == "mean":
                    if len(embeddings) == 1:
                        emb = embeddings[0]
                        # Mean pool over sequence dimension if 3D
                        return emb.mean(dim=1) if emb.dim() == 3 else emb
                    # For multiple embeddings with different dimensions, return list
                    # (probe will handle projection)
                    return embeddings
                else:
                    # For other aggregations, return first embedding
                    return embeddings[0]

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor (embeddings or logits).
                """
                # Actually use the layers so hooks can capture them
                # Flatten input if needed
                if x.dim() > 2:
                    x = x.view(x.shape[0], -1)
                # Pad or truncate to expected input size (128)
                if x.shape[1] < 128:
                    x = torch.nn.functional.pad(x, (0, 128 - x.shape[1]))
                elif x.shape[1] > 128:
                    x = x[:, :128]

                # Use layer1 and layer2 so hooks can capture them
                out = self.layer1(x)
                out = self.layer2(out)

                if self.return_features_only:
                    # Return 3D tensor (batch, seq, features) for sequence probes
                    # Expand to (batch, seq_len, features)
                    return out.unsqueeze(1).expand(-1, 10, -1)
                if self.num_classes is None:
                    return out
                return self.classifier(out)

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

    @pytest.fixture
    def base_model(self) -> ModelBase:
        """Create a base model for probe testing.

        Returns:
            ModelBase: A test model instance in embedding mode.
        """
        model = load_model("test_model", device="cpu", return_features_only=True)
        model.eval()
        return model

    def test_builds_linear_probe_online(self, base_model: ModelBase) -> None:
        """Test building linear probe in online mode."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            base_model=base_model,
            num_classes=10,
            device="cpu",
        )

        assert probe is not None
        assert hasattr(probe, "forward")
        assert hasattr(probe, "classifier")

        # Test forward pass
        dummy_input = torch.randn(2, 16000)
        probe.eval()
        with torch.no_grad():
            output = probe(dummy_input, padding_mask=None)
        assert output.shape == (2, 10)

    def test_builds_mlp_probe_online(self, base_model: ModelBase) -> None:
        """Test building MLP probe in online mode."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            target_layers=["last_layer"],
            aggregation="mean",
            hidden_dims=[64, 32],
            freeze_backbone=True,
            online_training=True,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            base_model=base_model,
            num_classes=5,
            device="cpu",
        )

        assert probe is not None
        assert hasattr(probe, "forward")

        # Test forward pass
        dummy_input = torch.randn(2, 16000)
        probe.eval()
        with torch.no_grad():
            output = probe(dummy_input, padding_mask=None)
        assert output.shape == (2, 5)

    def test_builds_linear_probe_offline(self) -> None:
        """Test building linear probe in offline mode (feature_mode)."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["backbone"],  # Not used in offline mode
            aggregation="none",
            freeze_backbone=True,
            online_training=False,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            input_dim=128,
            num_classes=10,
            device="cpu",
        )

        assert probe is not None
        assert hasattr(probe, "forward")

        # Test forward pass with pre-computed embeddings
        dummy_embeddings = torch.randn(2, 128)
        probe.eval()
        with torch.no_grad():
            output = probe(dummy_embeddings)
        assert output.shape == (2, 10)

    def test_builds_mlp_probe_offline(self) -> None:
        """Test building MLP probe in offline mode."""
        probe_config = ProbeConfig(
            probe_type="mlp",
            target_layers=["backbone"],
            aggregation="none",
            hidden_dims=[64],
            freeze_backbone=True,
            online_training=False,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            input_dim=256,
            num_classes=8,
            device="cpu",
        )

        assert probe is not None

        # Test forward pass
        dummy_embeddings = torch.randn(2, 256)
        probe.eval()
        with torch.no_grad():
            output = probe(dummy_embeddings)
        assert output.shape == (2, 8)

    def test_builds_probe_with_all_layers(self, base_model: ModelBase) -> None:
        """Test building probe with target_layers=['all']."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["all"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            base_model=base_model,
            num_classes=10,
            device="cpu",
        )

        assert probe is not None

        # Test forward pass
        dummy_input = torch.randn(2, 16000)
        probe.eval()
        with torch.no_grad():
            output = probe(dummy_input, padding_mask=None)
        assert output.shape == (2, 10)

    def test_raises_error_for_invalid_probe_type(self, base_model: ModelBase) -> None:
        """Test that ValueError is raised for invalid probe type.

        Note: ProbeConfig validates probe_type at Pydantic level, so we need to
        bypass that validation or test with a valid type that's not registered.
        We'll test the factory-level error by using a valid probe type but
        mocking the registry to not have it.
        """
        # Use a valid probe type but test that the factory handles missing registration
        # This is tested indirectly - ProbeConfig will reject invalid types at creation
        # The factory will handle unregistered but valid-looking types
        from avex.models.probes.utils.registry import _PROBE_CLASSES

        # Temporarily clear probe classes to simulate unregistered type
        original_classes = _PROBE_CLASSES.copy()
        _PROBE_CLASSES.clear()

        try:
            # Create config with valid type but unregistered class
            probe_config = ProbeConfig(
                probe_type="linear",  # Valid type but class not registered
                target_layers=["last_layer"],
                aggregation="mean",
                freeze_backbone=True,
                online_training=True,
            )

            with pytest.raises(ValueError, match="Probe class 'linear' is not registered"):
                build_probe_from_config(
                    probe_config=probe_config,
                    base_model=base_model,
                    num_classes=10,
                    device="cpu",
                )
        finally:
            # Restore original classes
            _PROBE_CLASSES.clear()
            _PROBE_CLASSES.update(original_classes)

    def test_raises_error_for_offline_without_input_dim(self) -> None:
        """Test that ValueError is raised for offline mode without input_dim."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["backbone"],
            aggregation="none",
            freeze_backbone=True,
            online_training=False,
        )

        with pytest.raises(ValueError, match="Must specify either"):
            build_probe_from_config(
                probe_config=probe_config,
                input_dim=None,  # This will cause a ValueError
                num_classes=10,
                device="cpu",
            )

    def test_raises_error_for_sequence_processing_incompatible_probe(self, base_model: ModelBase) -> None:
        """Test that ValueError is raised for incompatible input_processing."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            input_processing="sequence",  # Not compatible with linear probe
            freeze_backbone=True,
            online_training=True,
        )

        with pytest.raises(ValueError, match="Sequence input processing is not compatible with linear probe"):
            build_probe_from_config(
                probe_config=probe_config,
                base_model=base_model,
                num_classes=10,
                device="cpu",
            )

    def test_freezes_backbone_when_requested(self, base_model: ModelBase) -> None:
        """Test that backbone is frozen when freeze_backbone=True."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )

        probe = build_probe_from_config(
            probe_config=probe_config,
            base_model=base_model,
            num_classes=10,
            device="cpu",
        )

        # Check that backbone parameters are frozen
        for param in base_model.parameters():
            assert not param.requires_grad

        # Check that probe parameters are trainable
        probe_params = [p for p in probe.parameters() if p.requires_grad]
        assert len(probe_params) > 0

    def test_unfreezes_backbone_when_requested(self, base_model: ModelBase) -> None:
        """Test that backbone is unfrozen when freeze_backbone=False."""
        probe_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=False,
            online_training=True,
        )

        build_probe_from_config(
            probe_config=probe_config,
            base_model=base_model,
            num_classes=10,
            device="cpu",
        )

        # Check that backbone parameters are trainable
        backbone_params = [p for p in base_model.parameters() if p.requires_grad]
        assert len(backbone_params) > 0


class TestListModelLayers:
    """Test list_model_layers function."""

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

        # Register a test model class with layer discovery
        @register_model_class
        class TestModelClass(ModelBase):
            """Test model class with discoverable layers."""

            name = "test_model_type"

            def __init__(
                self,
                device: str,
                num_classes: int | None = None,
                audio_config: dict[str, Any] | AudioConfig | None = None,
                return_features_only: bool = False,
                **kwargs: object,
            ) -> None:
                if isinstance(audio_config, dict):
                    audio_config = AudioConfig(**audio_config)
                super().__init__(device=device, audio_config=audio_config)
                self.num_classes = num_classes
                self.return_features_only = return_features_only

                # Create layers for discovery
                self.layer1 = torch.nn.Linear(128, 64)
                self.layer2 = torch.nn.Linear(64, 32)
                if not return_features_only and num_classes is not None:
                    self.classifier = torch.nn.Linear(32, num_classes)
                self.to(device)

            def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
                """Forward pass.

                Returns:
                    torch.Tensor: Model output tensor (embeddings or logits).
                """
                # Actually use the layers so hooks can capture them
                if x.dim() > 2:
                    x = x.view(x.shape[0], -1)
                if x.shape[1] < 128:
                    x = torch.nn.functional.pad(x, (0, 128 - x.shape[1]))
                elif x.shape[1] > 128:
                    x = x[:, :128]

                out = self.layer1(x)
                out = self.layer2(out)

                if self.return_features_only:
                    return out.unsqueeze(1).expand(-1, 10, -1)
                return self.classifier(out) if hasattr(self, "classifier") else out

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

    def test_lists_layers_from_model_name(self) -> None:
        """Test listing layers using model name."""
        layers_info = list_model_layers("test_model", device="cpu")

        assert isinstance(layers_info, dict)
        assert "layers" in layers_info
        assert "last_layer" in layers_info
        assert "all" in layers_info
        assert "special_options" in layers_info

        assert isinstance(layers_info["layers"], list)
        assert len(layers_info["layers"]) > 0
        assert isinstance(layers_info["last_layer"], str)
        assert layers_info["last_layer"] in layers_info["layers"]
        assert layers_info["all"] == layers_info["layers"]
        assert layers_info["special_options"] == ["last_layer", "all"]

    def test_lists_layers_from_model_instance(self) -> None:
        """Test listing layers using model instance."""
        model = load_model("test_model", device="cpu", return_features_only=True)
        model.eval()

        layers_info = list_model_layers(model, device="cpu")

        assert isinstance(layers_info, dict)
        assert "layers" in layers_info
        assert len(layers_info["layers"]) > 0
        assert layers_info["last_layer"] in layers_info["layers"]

    def test_returns_correct_layer_structure(self) -> None:
        """Test that returned structure contains expected layers."""
        layers_info = list_model_layers("test_model", device="cpu")

        # Should discover layer1 and layer2 (but not classifier)
        layers = layers_info["layers"]
        assert len(layers) >= 2
        assert any("layer1" in layer for layer in layers)
        assert any("layer2" in layer for layer in layers)

        # last_layer should be one of the discovered layers
        assert layers_info["last_layer"] in layers

    def test_raises_error_for_unknown_model(self) -> None:
        """Test that ValueError is raised for unknown model name."""
        with pytest.raises(ValueError, match="not found in registry"):
            list_model_layers("nonexistent_model", device="cpu")

    def test_raises_error_for_non_modelbase_instance(self) -> None:
        """Test that ValueError is raised for non-ModelBase instance."""
        not_a_model = torch.nn.Linear(10, 5)

        with pytest.raises(ValueError, match="must be an instance of ModelBase"):
            list_model_layers(not_a_model, device="cpu")  # type: ignore[arg-type]

    def test_reuses_discovered_layers(self) -> None:
        """Test that layer discovery is cached and reused."""
        model = load_model("test_model", device="cpu", return_features_only=True)
        model.eval()

        # First call should discover layers
        layers_info1 = list_model_layers(model, device="cpu")
        layers1 = layers_info1["layers"]

        # Second call should reuse discovered layers (not rediscover)
        layers_info2 = list_model_layers(model, device="cpu")
        layers2 = layers_info2["layers"]

        # Should be the same (cached)
        assert layers1 == layers2
        assert len(layers1) == len(layers2)
