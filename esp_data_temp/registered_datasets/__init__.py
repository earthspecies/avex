"""Registered datasets package."""

from .registry import DatasetInfo, register_dataset, registry
from .animalspeak import AnimalSpeak

__all__ = ["DatasetInfo", "register_dataset", "registry", "AnimalSpeak"]
