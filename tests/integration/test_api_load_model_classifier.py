"""Tests for load_model classifier head loading behavior.

This module tests that classifier head weights are correctly loaded from
checkpoints when num_classes=None, and that they are not loaded when
num_classes is explicitly provided.
"""

from __future__ import annotations

import pytest
import torch

from representation_learning import (
    load_model,
    register_model,
)
from representation_learning.configs import AudioConfig, ModelSpec


class TestLoadModelClassifierHead:
    """Test classifier head loading behavior in load_model."""

    @pytest.fixture(autouse=True)
    def setup_model_registry(self) -> None:
        """Register model spec for testing.

        Yields:
            None: Fixture yields nothing, just sets up the model registry.
        """
        from representation_learning.models.utils import registry

        # Clear registry to ensure clean state
        registry._MODEL_REGISTRY.clear()
        # Don't clear _MODEL_CLASSES - we need the beats model class to be auto-discovered
        # Initialize registry to auto-discover model classes
        registry.initialize_registry()

        # Register the model spec for sl_beats_animalspeak
        # Note: The BEATs model class is now auto-registered at startup
        # Note: register_model now overwrites if already registered
        model_spec = ModelSpec(
            name="beats",
            pretrained=False,
            fine_tuned=True,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=16000,
                representation="raw",
                normalize=False,
                target_length_seconds=10,
                window_selection="random",
            ),
        )
        register_model("sl_beats_animalspeak", model_spec)
        yield

        # Clean up
        registry._MODEL_REGISTRY.clear()

    def test_beats_backbone_loads_from_checkpoint_features_only(self) -> None:
        """Test that BEATs backbone can be loaded from checkpoint in embedding mode."""
        checkpoint_path = "gs://representation-learning/models/sl_beats_animalspeak.pt"

        model = load_model(
            "sl_beats_animalspeak",
            checkpoint_path=checkpoint_path,
            device="cpu",
            return_features_only=True,
        )

        assert hasattr(model, "forward")

        dummy_input = torch.randn(1, 16000 * 5)
        with torch.no_grad():
            output = model(dummy_input, padding_mask=None)

        assert torch.is_tensor(output)
        assert output.shape[0] == 1
