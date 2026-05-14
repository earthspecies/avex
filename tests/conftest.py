from __future__ import annotations

import os
import random
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def preserve_model_registry_state() -> Generator[None, None, None]:
    """Keep registry-mutating tests from leaking global state."""
    from avex.models.utils import registry

    model_registry = registry._MODEL_REGISTRY.copy()
    model_classes = registry._MODEL_CLASSES.copy()
    official_models_pkg = registry._OFFICIAL_MODELS_PKG

    yield

    registry._MODEL_REGISTRY.clear()
    registry._MODEL_REGISTRY.update(model_registry)
    registry._MODEL_CLASSES.clear()
    registry._MODEL_CLASSES.update(model_classes)
    registry._OFFICIAL_MODELS_PKG = official_models_pkg


@pytest.fixture(autouse=True, scope="session")
def set_global_determinism() -> None:
    """Set deterministic flags and seeds for reproducible tests."""
    seed = int(os.environ.get("PYTEST_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture()
def torch_no_grad() -> Generator[None, None, None]:
    """Context manager fixture to ensure no grad during test scopes when used."""
    with torch.no_grad():
        yield
