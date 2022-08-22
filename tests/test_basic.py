from mailbox import _mboxMMDFMessage
from re import L
import pytest
import logging
import pythonrlt
import pandas as pd
import faulthandler
import numpy as np
import pandas as pd

faulthandler.enable()


LOGGER = logging.getLogger(__name__)


def test_binding():
    LOGGER.debug("Test pybind11 with a dummy `add` function.")
    assert pythonrlt.add(1, 2) == 3


def test_data_loading():
    # LOGGER.debug(f"train_csv = {train_csv}")
    # LOGGER.debug(f"test_csv = {test_csv}")

    trainX = pd.read_csv("./tests/data/trainX.csv")
    trainY = pd.read_csv("./tests/data/trainY.csv")

    LOGGER.debug("{} {}".format(str(trainY.shape), str(trainY.columns)))
    assert len(trainX) == len(trainY)

    # TODO(ofey404): Use those data as you wish!
    #                Definition of those fixtures are in tests/conftest.py
    #                or check https://docs.pytest.org/en/6.2.x/fixture.html


def test_very_small_dataset():
    trainX = np.array([[10, 12], [9, 8], [3, 7], [-1, 3],
                      [12, 13], [0, 0], [9, 9]]).astype("double")
    trainY = np.array([11, 8.5, 4, 0, 14, -1, 11]).astype("double")
    fit = pythonrlt.pythonRegWithGivenXYReturnList(trainX, trainY, 2)


def test_simulated_data_crosscheck_R():
    # LOGGER.debug(f"train_csv = {train_csv}")
    # LOGGER.debug(f"test_csv = {test_csv}")

    trainX = pd.read_csv("./tests/data/trainX.csv")
    trainX_mat = trainX.iloc[:, 1:].astype("double")
    trainY = pd.read_csv("./tests/data/trainY.csv")['x'].astype("double")
    fit = pythonrlt.pythonRegWithGivenXYReturnList(trainX, trainY, 2)
    assert len(trainX) == len(trainY)
