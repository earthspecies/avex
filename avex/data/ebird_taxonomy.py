"""eBird species taxonomy bundled with the package.

Provides a mapping from eBird species codes (e.g. ``"ostric2"``) to common
and scientific names.  Each release is stored in a separate JSON file so models
always resolve labels against the taxonomy version they were trained with.

Usage::

    from avex.data.ebird_taxonomy import load

    tax = load("v2021")
    tax["ostric2"]  # {"common_name": "Common Ostrich", "sci_name": "Struthio camelus"}
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Literal

EbirdTaxonomyVersion = Literal["v2021", "v2025"]

_VERSION_FILES: dict[EbirdTaxonomyVersion, str] = {
    "v2021": "ebird_taxonomy_v2021.json",
    "v2025": "ebird_taxonomy_v2025.json",
}


@lru_cache(maxsize=len(_VERSION_FILES))
def load(version: EbirdTaxonomyVersion) -> dict[str, dict[str, str]]:
    """Return the eBird taxonomy mapping for a specific release.

    Parameters
    ----------
    version
        Taxonomy release to load. Use ``"v2021"`` for BirdSet XCL / AudioProtoPNet
        checkpoints and ``"v2025"`` for the birdcode SED checkpoint.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{ebird_code: {"common_name": ..., "sci_name": ...}}``. The mapping is
        cached (``lru_cache``) and shared across callers; treat it as read-only
        and copy before mutating.

    Raises
    ------
    ValueError
        If ``version`` is not a supported taxonomy release.
    FileNotFoundError
        If the bundled JSON for ``version`` is missing from the package.
    """
    json_name = _VERSION_FILES.get(version)
    if json_name is None:
        supported = ", ".join(sorted(_VERSION_FILES))
        raise ValueError(f"Unsupported eBird taxonomy version {version!r}. Supported: {supported}.")

    pkg = resources.files(__name__.rsplit(".", 1)[0])  # avex.data
    json_file = pkg / json_name
    if not json_file.is_file():
        raise FileNotFoundError(f"Bundled eBird taxonomy file not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as f:
        return json.load(f)
