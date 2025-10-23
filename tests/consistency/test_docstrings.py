"""Tests for checking the docstrings of functions and classes.

Header info specific to ESP
"""

from tests.utils.check_docstrings import check_docstrings


def test_docstrings_exist(base_folder: str) -> None:
    """Check that all class and functions contain a docstring.
    Numpy-style is used.

    Arguments
    ---------
    base_folder: str, optional
        Path to the base folder where this function is executed.
    folders_to_check: list[str], optional
        Folders name that must be checked, by default, all of them.

    """
    # Only check the representation_learning package, not dependencies
    assert check_docstrings(base_folder, ["representation_learning"])
