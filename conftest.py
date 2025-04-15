"""This is used to parametrise a few things w.r.t testing with
pytest

Header info specific to ESP
"""

import pytest


# ESP only has CPU testing for now?
def pytest_addoption(parser: pytest.Parser) -> None:
    """This functionis used to decorate automatically other functions
    with arguments that should be passed to all unit tests functions.
    For now we only support the device argument.
    """
    parser.addoption("--base_folder", action="store", default=".")
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """This is called for every test. Only get/set command line arguments
    if the argument is specified in the list of test "fixturenames".
    """

    option_value = metafunc.config.option.base_folder
    if "base_folder" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("base_folder", [option_value])

    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])
