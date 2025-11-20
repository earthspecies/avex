"""Data module for representation learning."""

# Import transforms to ensure they are registered
# Import VFPA dataset so it registers with esp-data when package loads
from . import (
    transforms,  # noqa: F401
    vfpa,  # noqa: F401
)
