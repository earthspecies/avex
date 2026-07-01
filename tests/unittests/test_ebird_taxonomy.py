"""Tests for versioned eBird taxonomy lookups."""

from __future__ import annotations

from pathlib import Path

import pytest

from avex.data import ebird_taxonomy as et


@pytest.mark.parametrize("version", ["v2021", "v2025"])
def test_load_returns_mapping(version: et.EbirdTaxonomyVersion) -> None:
    tax = et.load(version)
    assert isinstance(tax, dict)
    assert tax["ostric2"]["common_name"] == "Common Ostrich"
    assert tax["ostric2"]["sci_name"] == "Struthio camelus"


def test_versions_overlap_with_taxonomic_drift() -> None:
    v2021 = et.load("v2021")
    v2025 = et.load("v2025")
    conflicts = [code for code in v2021 if code in v2025 and v2021[code] != v2025[code]]
    assert conflicts, "expected some taxonomic drift between v2021 and v2025"
    assert v2021["grskiw1"]["sci_name"] == "Apteryx haastii"
    assert v2025["grskiw1"]["sci_name"] == "Apteryx maxima"


def test_birdset_conflict_codes_follow_v2021_not_v2025() -> None:
    """BirdSet XCL uses eBird v2021 names where v2021 and v2025 diverge."""
    tax21 = et.load("v2021")
    tax25 = et.load("v2025")
    assert tax21["grskiw1"]["sci_name"] == "Apteryx haastii"
    assert tax25["grskiw1"]["sci_name"] == "Apteryx maxima"
    assert tax21["martea1"]["common_name"] == "Marbled Teal"
    assert tax25["martea1"]["common_name"] == "Marbled Duck"


def test_load_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="Unsupported eBird taxonomy version"):
        et.load("v2099")  # type: ignore[arg-type]


def test_bundled_json_files_exist() -> None:
    data_dir = Path(et.__file__).resolve().parent
    assert (data_dir / "ebird_taxonomy_v2021.json").is_file()
    assert (data_dir / "ebird_taxonomy_v2025.json").is_file()
    assert not (data_dir / "ebird_taxonomy.json").exists()
