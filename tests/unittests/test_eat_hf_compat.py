"""Tests for the EAT HF transformers >= 5 shim that do not hit the Hugging Face Hub."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from avex.models import eat_hf


class _FakeRemoteModel:
    """Stands in for the remote EATModel class."""

    calls_post_init = False

    def __init__(self, config: object) -> None:
        self.post_init_count = 0
        if self.calls_post_init:
            self.post_init()

    def post_init(self) -> None:
        self.post_init_count += 1

    @classmethod
    def from_pretrained(cls, model_name: str, config: object) -> "_FakeRemoteModel":
        return cls(config)


class _FakeRemoteModelFixed(_FakeRemoteModel):
    """Remote class after upstream adds post_init() to __init__."""

    calls_post_init = True


def _load_with_fake(remote_cls: type, auto_map: dict | None) -> object:
    config = SimpleNamespace(auto_map=auto_map)
    with (
        patch.object(eat_hf.AutoConfig, "from_pretrained", return_value=config),
        patch.object(eat_hf, "get_class_from_dynamic_module", return_value=remote_cls) as get_cls,
    ):
        model = eat_hf._load_backbone("worstchan/EAT-base_epoch30_pretrain")
    get_cls.assert_called_once_with("modeling_eat.EATModel", "worstchan/EAT-base_epoch30_pretrain")
    return model


@pytest.mark.parametrize("remote_cls", [_FakeRemoteModel, _FakeRemoteModelFixed])
def test_shim_calls_post_init_exactly_once(remote_cls: type) -> None:
    """post_init() runs once whether or not upstream calls it."""
    model = _load_with_fake(remote_cls, {"AutoModel": "modeling_eat.EATModel"})
    assert model.post_init_count == 1
    assert type(model).__name__ == remote_cls.__name__


@pytest.mark.parametrize("auto_map", [None, {}, {"AutoConfig": "configuration_eat.EATConfig"}])
def test_non_remote_code_repo_raises(auto_map: dict | None) -> None:
    """A repo without an AutoModel remote class gives a clear error."""
    config = SimpleNamespace(auto_map=auto_map)
    with (
        patch.object(eat_hf.AutoConfig, "from_pretrained", return_value=config),
        pytest.raises(ValueError, match="not a remote-code EAT repository"),
    ):
        eat_hf._load_backbone("some/other-repo")
