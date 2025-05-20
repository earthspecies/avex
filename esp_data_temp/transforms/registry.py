from typing import Annotated, Any, Callable, Union, get_args

import pandas as pd
from pydantic import BaseModel, Field

__all__ = [
    "register_transform",
    "transform_from_config",
    "RegisteredTransformConfigs",
]

RegisteredTransformConfigs = Any

_TRANSFORM_CONFIG_REGISTRY: dict[str, type[BaseModel]] = {}
_TRANSFORM_FACTORY_REGISTRY: dict[type[BaseModel], type] = {}


def register_transform(config_class: type[BaseModel], transform_class: type) -> None:
    """Register a transform configuration class and its corresponding transform class.

    This is used to (1) create a union type of all available transform configurations
    which is used by pydantic for validation and type checking (2) pick up the correct
    factory method when instantiating a transform object from the configuration and (3)
    ensure that the transform type is unique.

    Parameters
    ----------
    config_class : type[BaseModel]
        The pydantic configuration class that inherits from `BaseModel`. This class
        should have a unique `type` attribute which is used to identify the transform
        type.
    transform_class : type
        The transform class that implements the actual transformation logic. This class
        should have a `from_config` method which takes the configuration class as an
        argument and returns an instance of the transform class.

    Raises
    ------
    ValueError
        If the transform type is already registered, a ValueError is raised.
        If the transform type is not a single value, a ValueError is raised.

    AttributeError
        If the transform class does not have a `from_config` method, an AttributeError
        is raised.
    """

    def _rebuild_union_type() -> None:
        global RegisteredTransformConfigs
        RegisteredTransformConfigs = Annotated[
            Union[tuple(_TRANSFORM_CONFIG_REGISTRY.values())],
            Field(discriminator="type"),
        ]

    if "type" not in config_class.model_fields:
        raise ValueError(
            f"Transform configuration class '{config_class.__name__}' "
            "does not have a 'type' field."
        )

    type_vals = get_args(config_class.model_fields["type"].annotation)
    if len(type_vals) == 1:
        v = type_vals[0]

        if v in _TRANSFORM_CONFIG_REGISTRY:
            raise ValueError(
                f"Transform type '{v}' is already registered. "
                "Please use a unique type name."
            )

        _TRANSFORM_CONFIG_REGISTRY[v] = config_class
        _rebuild_union_type()
    else:
        raise ValueError(f"Transform type is not a single value: {type_vals}")

    if not hasattr(transform_class, "from_config"):
        raise AttributeError(
            f"Class '{transform_class.__name__}' does not have a 'from_config' method."
            "This method should take the configuration class as an argument and return"
            " an instance of the transform class."
        )

    _TRANSFORM_FACTORY_REGISTRY[config_class] = transform_class


def transform_from_config(
    cfg: RegisteredTransformConfigs,
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]:
    """Create a callable transform object from a configuration object.

    Parameters
    ----------
    cfg : RegisteredTransformConfigs
        The configuration object that specifies the transform type and parameters.

    Returns
    -------
    Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]
        A callable that takes a DataFrame as input and returns a tuple of the
        transformed DataFrame and a dictionary of metadata.
    """

    return _TRANSFORM_FACTORY_REGISTRY[type(cfg)].from_config(cfg)
