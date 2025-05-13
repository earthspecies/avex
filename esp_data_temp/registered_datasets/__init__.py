"""Registered datasets package."""

from .registry import DatasetInfo, register_dataset, registry

__all__ = ["DatasetInfo", "register_dataset", "registry"]
