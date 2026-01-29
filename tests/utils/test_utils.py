"""Shared test utilities for model testing.

This module provides reusable fixtures and utilities for testing models,
particularly for hook management and cleanup.
"""

from __future__ import annotations

from typing import Callable, Generator

import pytest

from avex.models.base_model import ModelBase


def create_cleanup_hooks_fixture(
    model_fixture_name: str = "model",
    hook_layers: list[str] | None = None,
) -> Callable:
    """Create an autouse fixture that cleans up model hooks after each test.

    This fixture ensures that hooks are properly deregistered after each test
    to prevent test interference. Optionally, it can also re-register hooks
    before each test to ensure consistent state.

    Args:
        model_fixture_name: Name of the model fixture to clean up hooks for.
            Defaults to "model".
        hook_layers: Optional list of layer names to re-register hooks for
            before each test. If None, only cleanup is performed. If provided,
            hooks are re-registered before each test and cleaned up after.

    Returns:
        A pytest fixture function that can be used as a decorator.

    Example:
        ```python
        class TestMyModel:
            @pytest.fixture(scope="class")
            def model(self) -> MyModel:
                return MyModel()

            cleanup_hooks = create_cleanup_hooks_fixture(
                model_fixture_name="model",
                hook_layers=["layer1", "layer2"],
            )
        ```

    Note:
        The fixture uses `autouse=True` so it will automatically run for all
        tests in the class. It uses `request.getfixturevalue()` to access the
        model fixture dynamically.
    """

    @pytest.fixture(autouse=True)
    def cleanup_hooks(request: pytest.FixtureRequest) -> Generator[None, None, None]:
        """Clean up model hooks after each test.

        Args:
            request: Pytest request object to access test fixtures.
        """
        # Re-register hooks before test if hook_layers provided
        try:
            if (
                hook_layers is not None
                and hasattr(request, "fixturenames")
                and model_fixture_name in request.fixturenames
            ):
                model = request.getfixturevalue(model_fixture_name)
                if hasattr(model, "register_hooks_for_layers"):
                    model.register_hooks_for_layers(hook_layers)
        except (AttributeError, TypeError):
            # request is not a FixtureRequest (e.g., test class instance)
            pass

        yield

        # Clean up hooks after test
        try:
            if hasattr(request, "fixturenames") and model_fixture_name in request.fixturenames:
                model = request.getfixturevalue(model_fixture_name)
                if hasattr(model, "deregister_all_hooks"):
                    model.deregister_all_hooks()
        except (AttributeError, TypeError):
            # request is not a FixtureRequest (e.g., test class instance)
            pass

    return cleanup_hooks


def cleanup_model_hooks(model: ModelBase) -> None:
    """Clean up hooks for a model instance.

    This is a simple utility function that can be called directly if you
    don't need the fixture pattern.

    Args:
        model: Model instance to clean up hooks for.
    """
    if hasattr(model, "deregister_all_hooks"):
        model.deregister_all_hooks()
