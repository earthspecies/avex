"""Tests for the BirdMAE wrapper that do not hit the Hugging Face Hub."""

from importlib import resources
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from avex.configs import ModelSpec
from avex.models import birdmae
from avex.models.utils import registry


class _FakeRemoteModel:
    """Stands in for the remote BirdMAEModel class."""

    calls_post_init = False

    def __init__(self, config: object) -> None:
        self.post_init_count = 0
        if self.calls_post_init:
            self.post_init()

    def post_init(self) -> None:
        self.post_init_count += 1


class _FakeRemoteModelFixed(_FakeRemoteModel):
    """Remote class after upstream adds post_init() to __init__."""

    calls_post_init = True


def _load_with_fake(remote_cls: type, model_id: str, revision: str | None) -> tuple[type, object, dict]:
    config = SimpleNamespace(auto_map={"AutoModel": "modeling_bird_mae.BirdMAEModel"})
    seen: dict = {}

    def fake_get_class(class_ref: str, repo_id: str, revision: str | None = None) -> type:
        seen["get_class"] = (class_ref, repo_id, revision)
        return remote_cls

    def fake_config(repo_id: str, trust_remote_code: bool, revision: str | None = None) -> object:
        seen["config"] = (repo_id, trust_remote_code, revision)
        return config

    with (
        patch("transformers.AutoConfig.from_pretrained", side_effect=fake_config),
        patch("transformers.dynamic_module_utils.get_class_from_dynamic_module", side_effect=fake_get_class),
    ):
        cls, cfg = birdmae._load_model_class(model_id, revision)
    return cls, cfg, seen


@pytest.mark.parametrize("remote_cls", [_FakeRemoteModel, _FakeRemoteModelFixed])
def test_shim_calls_post_init_exactly_once(remote_cls: type) -> None:
    """post_init() runs once whether or not upstream calls it."""
    cls, cfg, _ = _load_with_fake(remote_cls, birdmae.DEFAULT_MODEL_ID, None)
    model = cls(cfg)
    assert model.post_init_count == 1
    assert cls.__name__ == remote_cls.__name__


def test_shim_passes_revision_to_hub() -> None:
    """The pinned revision is used for both config and remote code."""
    revision = birdmae._PINNED_REVISIONS[birdmae.DEFAULT_MODEL_ID]
    _, _, seen = _load_with_fake(_FakeRemoteModel, birdmae.DEFAULT_MODEL_ID, revision)
    assert seen["config"] == (birdmae.DEFAULT_MODEL_ID, True, revision)
    assert seen["get_class"] == ("modeling_bird_mae.BirdMAEModel", birdmae.DEFAULT_MODEL_ID, revision)


def test_known_repos_are_pinned() -> None:
    """Both registered Bird-MAE repos have a full commit sha pinned."""
    for model_id in ("DBD-research-group/Bird-MAE-Base", "DBD-research-group/Bird-MAE-huge"):
        revision = birdmae._PINNED_REVISIONS[model_id]
        assert len(revision) == 40
        int(revision, 16)


@pytest.mark.parametrize(
    "name, model_id",
    [
        ("bird_mae_base", "DBD-research-group/Bird-MAE-Base"),
        ("bird_mae_huge", "DBD-research-group/Bird-MAE-huge"),
    ],
)
def test_registry_entries(name: str, model_id: str) -> None:
    """Bird-MAE models are packaged as registry YAMLs and point at a pinned repo.

    Reads the packaged YAML and runs class discovery directly, since other
    tests clear the global registry.
    """
    yaml_file = resources.files(registry._OFFICIAL_MODELS_PKG) / f"{name}.yml"
    with resources.as_file(yaml_file) as yaml_path:
        spec = registry.load_model_spec_from_yaml(yaml_path)
    assert spec.name == "birdmae"
    assert spec.model_id == model_id
    assert model_id in birdmae._PINNED_REVISIONS
    assert spec.audio_config.sample_rate == 32000
    assert registry._discover_model_classes()["birdmae"] is birdmae.Model


def test_model_spec_model_id_defaults_to_none() -> None:
    """ModelSpec must not leak a model-specific hub id into other models."""
    assert ModelSpec(name="birdmae").model_id is None


@pytest.mark.parametrize("length, expected", [(498, 512), (512, 512), (600, 512)])
def test_fit_time_axis_pads_and_crops(length: int, expected: int) -> None:
    """Time axis is padded with the batch minimum or cropped to target_length, regardless of batch size."""
    import torch

    features = torch.rand(513, 1, length, 128) + 1.0
    features[0, 0, 0, 0] = -3.0
    out = birdmae._fit_time_axis(features, 512)
    assert out.shape == (513, 1, expected, 128)
    if length < 512:
        assert torch.all(out[:, :, length:, :] == -3.0)
        assert torch.equal(out[:, :, :length, :], features)
    else:
        assert torch.equal(out, features[:, :, :512, :])
