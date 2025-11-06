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
    register_model_class,
    unregister_model,
    unregister_model_class,
)
from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.beats_model import Model as BeatsModel


class TestLoadModelClassifierHead:
    """Test classifier head loading behavior in load_model."""

    @pytest.fixture(autouse=True)
    def setup_model_class_and_registry(self) -> None:
        """Register BEATs model class and model spec for testing.

        Yields:
            None: Fixture yields nothing, just sets up the model class and registry.
        """
        # Register the BEATs model class so build_model_from_spec can find it
        # The model spec uses "beats" as the name, so we need to register it as "beats"
        # Set the name attribute temporarily if needed
        original_name = getattr(BeatsModel, "name", None)
        BeatsModel.name = "beats"  # type: ignore[attr-defined]
        try:
            register_model_class(BeatsModel)
            # Register the model spec for sl_beats_animalspeak
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
        finally:
            # Cleanup: unregister and restore original name
            unregister_model_class("beats")
            unregister_model("sl_beats_animalspeak")
            if original_name is not None:
                BeatsModel.name = original_name  # type: ignore[attr-defined]
            elif hasattr(BeatsModel, "name"):
                delattr(BeatsModel, "name")

    def test_beats_classifier_weights_loaded_when_num_classes_none(self) -> None:
        """Test that BEATs classifier weights are loaded when num_classes=None.

        This verifies the new behavior: when num_classes is None and extracted
        from checkpoint, the classifier head weights from the checkpoint should
        be preserved.
        """
        # Use the actual checkpoint from the config
        checkpoint_path = "gs://representation-learning/models/sl_beats_animalspeak.pt"

        # Load model with num_classes=None (should extract from checkpoint
        # and keep classifier weights)
        # Pass checkpoint_path directly to load_model (no longer using registry)
        loaded_model = load_model(
            "sl_beats_animalspeak",
            num_classes=None,  # Should extract from checkpoint
            checkpoint_path=checkpoint_path,
            device="cpu",
        )

        # Verify model has classifier
        assert hasattr(loaded_model, "classifier"), "Model should have classifier"

        # Verify num_classes was extracted correctly
        # (actual checkpoint has 12279 classes)
        actual_num_classes = loaded_model.classifier.weight.shape[0]
        assert actual_num_classes == 12279, (
            f"Expected 12279 classes from checkpoint, got {actual_num_classes}"
        )

        # Load the checkpoint directly to compare weights
        from esp_data.io import anypath

        ckpt_path = anypath(checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get classifier weights from checkpoint
        if isinstance(checkpoint, dict):
            # Handle model_state_dict wrapper
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Find classifier weights in checkpoint
            checkpoint_classifier_weight = None
            checkpoint_classifier_bias = None
            for key, value in state_dict.items():
                # Remove common prefixes
                clean_key = key
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                elif clean_key.startswith("model."):
                    clean_key = clean_key[6:]

                if clean_key == "classifier.weight":
                    checkpoint_classifier_weight = value
                elif clean_key == "classifier.bias":
                    checkpoint_classifier_bias = value

            # Verify classifier weights match the checkpoint (they should be kept)
            if checkpoint_classifier_weight is not None:
                assert torch.allclose(
                    loaded_model.classifier.weight,
                    checkpoint_classifier_weight,
                    atol=1e-6,
                ), (
                    "BEATs classifier weights should match checkpoint "
                    "when num_classes=None"
                )

            if checkpoint_classifier_bias is not None:
                assert torch.allclose(
                    loaded_model.classifier.bias,
                    checkpoint_classifier_bias,
                    atol=1e-6,
                ), "BEATs classifier bias should match checkpoint when num_classes=None"

    def test_beats_classifier_weights_not_loaded_when_num_classes_provided(
        self,
    ) -> None:
        """Test that BEATs classifier weights are NOT loaded when
        num_classes is explicit.

        This verifies the existing behavior: when num_classes is explicitly provided,
        the classifier head should be randomly initialized (for fine-tuning scenarios).
        """
        # Use the actual checkpoint from the config
        checkpoint_path = "gs://representation-learning/models/sl_beats_animalspeak.pt"

        # Load checkpoint to get original classifier weights for comparison
        from esp_data.io import anypath

        ckpt_path = anypath(checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get classifier weights from checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            checkpoint_classifier_weight = None
            for key, value in state_dict.items():
                clean_key = key
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                elif clean_key.startswith("model."):
                    clean_key = clean_key[6:]

                if clean_key == "classifier.weight":
                    checkpoint_classifier_weight = value
                    break

            # Checkpoint has 12279 classes, we'll use a different number
            new_num_classes = 10
            loaded_model = load_model(
                "sl_beats_animalspeak",
                num_classes=new_num_classes,  # Explicit num_classes
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

            # Verify classifier shape matches new num_classes
            assert loaded_model.classifier.weight.shape[0] == new_num_classes
            assert loaded_model.classifier.bias.shape[0] == new_num_classes

            # Verify classifier weights are DIFFERENT from checkpoint
            # (different shape confirms different initialization)
            assert checkpoint_classifier_weight is not None
            assert (
                loaded_model.classifier.weight.shape
                != checkpoint_classifier_weight.shape
            )
            assert (
                loaded_model.classifier.weight.shape[1]
                == checkpoint_classifier_weight.shape[1]
            ), "Embedding dimension should match"

    def test_beats_classifier_weights_not_loaded_same_num_classes_explicit(
        self,
    ) -> None:
        """Test BEATs behavior when num_classes matches checkpoint but is explicit.

        Even if num_classes matches the checkpoint, if it's explicitly provided,
        classifier weights should NOT be loaded (for fine-tuning scenarios where
        you want a fresh random head).
        """
        # Use the actual checkpoint from the config
        checkpoint_path = "gs://representation-learning/models/sl_beats_animalspeak.pt"

        # Load checkpoint to get original classifier weights
        from esp_data.io import anypath

        ckpt_path = anypath(checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get classifier weights from checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            checkpoint_classifier_weight = None
            checkpoint_classifier_bias = None
            for key, value in state_dict.items():
                clean_key = key
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                elif clean_key.startswith("model."):
                    clean_key = clean_key[6:]

                if clean_key == "classifier.weight":
                    checkpoint_classifier_weight = value
                elif clean_key == "classifier.bias":
                    checkpoint_classifier_bias = value

            # Load model with explicit num_classes (same as checkpoint: 12279)
            # Even though it matches, classifier should NOT be loaded (random init)
            loaded_model = load_model(
                "sl_beats_animalspeak",
                num_classes=12279,  # Explicit, even though it matches checkpoint
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

            # Verify classifier weights are DIFFERENT from checkpoint
            # (they should be randomly initialized, not loaded)
            # Note: There's a small chance of collision with random init,
            # but it's extremely unlikely
            assert checkpoint_classifier_weight is not None
            assert checkpoint_classifier_bias is not None

            weights_different = not torch.allclose(
                loaded_model.classifier.weight,
                checkpoint_classifier_weight,
                atol=1e-6,
            )
            bias_different = not torch.allclose(
                loaded_model.classifier.bias,
                checkpoint_classifier_bias,
                atol=1e-6,
            )

            # At least one should be different (weights or bias)
            assert weights_different or bias_different, (
                "BEATs classifier weights should be randomly initialized "
                "when num_classes is explicit"
            )
