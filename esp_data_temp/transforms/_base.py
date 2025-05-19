from typing import Annotated, Any, Callable, Union, get_args

import pandas as pd
from pydantic import BaseModel, Field

RegisteredTransformConfigs = Any
_TRANSFORM_REGISTRY: dict[str, type[BaseModel]] = {}
_TRANSFORM_FACTORY_REGISTRY: dict[type[BaseModel], type] = {}


# class TransformModel(BaseModel):
#     """
#     Base class for all transform configurations.

#     All transform configurations should inherit from this class and define a unique
#     `type` attribute. The `type` attribute can be anything (although it makes sense to
#     choose a descriptive name) as long as it's unique. This base class does the
#     registration of subclasses in the `_TRANSFORM_REGISTRY` dictionary and checks that
#     it is unique. It also generates `RegisteredTransformConfigs` type alias which is a
#     union of all registered transform types, allowing for easy validation and type
#     checking.
#     """

#     # TODO (milad) I wonder if this can be done with a simple decorator. Decorators are
#     # also called when a class is defined (and not when they're instantiated). Having
#     # a base class allows us to define the common "type" field but my type checker is
#     # complaining that you can't have the base hint as "str" while child classes are
#     # using Literal["..."].

#     type: str

#     @classmethod
#     def __pydantic_init_subclass__(cls, **kwargs) -> None:
#         """
#         This method is called when a subclass of TransformModel is created. It registers
#         the subclass in the `_TRANSFORM_REGISTRY` dictionary using the `type` attribute
#         as the key. It also rebuilds the `RegisteredTransformConfigs` union type to
#         include the new subclass, which is used for instantiating a _list_ of transforms

#         Raises
#         ------
#         ValueError
#             If the type is already registered, a ValueError is raised.
#         """

#         # See this issue below to understand we need `__pydantic_init_subclass__`
#         # instead of `__init_subclass__`:
#         # https://github.com/pydantic/pydantic/issues/6854

#         # TODO (milad) type checkers do not like dynamic types and complain about
#         # RegisteredTransformConfigs. This type is obviously useful for Pydantic checks
#         # but investigate if there's a static option.

#         type_vals = get_args(cls.model_fields["type"].annotation)
#         if len(type_vals) == 1:
#             v = type_vals[0]

#             if v in _TRANSFORM_REGISTRY:
#                 raise ValueError(
#                     f"Transform type '{v}' is already registered. "
#                     "Please use a unique type name."
#                 )

#             _TRANSFORM_REGISTRY[v] = cls
#             TransformModel._rebuild_union_type()

#     @classmethod
#     def _rebuild_union_type(cls) -> None:
#         global RegisteredTransformConfigs
#         RegisteredTransformConfigs = Annotated[
#             Union[tuple(_TRANSFORM_REGISTRY.values())], Field(discriminator="type")
#         ]


def register_transform(config_class: type[BaseModel], transform_class: type):
    """
    Register a transform configuration class and its corresponding transform class.
    """

    def _rebuild_union_type() -> None:
        global RegisteredTransformConfigs
        RegisteredTransformConfigs = Annotated[
            Union[tuple(_TRANSFORM_REGISTRY.values())], Field(discriminator="type")
        ]

    # TODO (milad) check that type exists and complain if it doesn't?

    type_vals = get_args(config_class.model_fields["type"].annotation)
    if len(type_vals) == 1:
        v = type_vals[0]

        if v in _TRANSFORM_REGISTRY:
            raise ValueError(
                f"Transform type '{v}' is already registered. "
                "Please use a unique type name."
            )

        _TRANSFORM_REGISTRY[v] = config_class
        _rebuild_union_type()
    else:
        raise ValueError(f"Transform type '{type_vals}' is not a single value.")

    # TODO (milad) assert that class has from_config factory method?

    _TRANSFORM_FACTORY_REGISTRY[config_class] = transform_class


def transform_from_config(
    cfg: RegisteredTransformConfigs,
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]:
    return _TRANSFORM_FACTORY_REGISTRY[type(cfg)].from_config(cfg)
