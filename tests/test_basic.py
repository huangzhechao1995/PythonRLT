import pytest
import logging
import pythonrlt

LOGGER = logging.getLogger(__name__)

def test_binding():
    LOGGER.debug("Test pybind11 with a dummy `add` function.")
    assert pythonrlt.add(1, 2) == 3


def test_example(train_csv, test_csv):
    LOGGER.debug(f"train_csv = {train_csv}")
    LOGGER.debug(f"test_csv = {test_csv}")

    # TODO(ofey404): Use those data as you wish!
    #                Definition of those fixtures are in tests/conftest.py
    #                or check https://docs.pytest.org/en/6.2.x/fixture.html
