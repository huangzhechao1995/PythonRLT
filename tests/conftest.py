import pytest
import logging

from pathlib import Path

import pandas as pd 

TEST_DIR = Path(__file__).resolve().parent

LOGGER = logging.getLogger(__name__)


@pytest.fixture()
def trainY_csv():
    data = pd.read_csv(TEST_DIR / "data/trainY.csv")
    yield data.astype("double")

@pytest.fixture()
def trainX_csv():
    data = pd.read_csv(TEST_DIR / "data/trainX.csv")
    yield data.astype("double")