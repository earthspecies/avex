"""eBird species taxonomy bundled with the package.

Provides a mapping from eBird species codes (e.g. ``"ostric2"``) to common
and scientific names.  The JSON file covers the full eBird taxonomy
(v2021 + v2025) — 18 130 taxa — which is a superset of the 9 736 species used
by BirdSet XCL.

Usage::

    from avex.data.ebird_taxonomy import load
    tax = load()
    tax["ostric2"]  # {"common_name": "Common Ostrich", "sci_name": "Struthio camelus"}
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib import resources

logger = logging.getLogger(__name__)

_JSON_NAME = "ebird_taxonomy.json"


@lru_cache(maxsize=1)
def load() -> dict[str, dict[str, str]]:
    """Return the eBird taxonomy mapping.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{ebird_code: {"common_name": ..., "sci_name": ...}}``
    """
    pkg = resources.files(__name__.rsplit(".", 1)[0])  # avex.data
    json_file = pkg / _JSON_NAME
    with json_file.open("r", encoding="utf-8") as f:
        return json.load(f)
