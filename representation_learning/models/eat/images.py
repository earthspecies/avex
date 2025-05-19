"""Compatibility wrapper exporting symbols from .image.

Data2VecMultiModel and other modules historically imported the module as
``from .images import ...`` while the actual implementation lives in
``image.py``.  This stub re-exports everything so both import styles work
without modifying upstream code.
"""

from .image import *  # noqa: F403,F401
