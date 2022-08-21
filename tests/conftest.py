import pytest
import logging

from pathlib import Path

import pandas as pd 

TEST_DIR = Path(__file__).resolve().parent

LOGGER = logging.getLogger(__name__)


@pytest.fixture()
def test_csv():
    data = pd.read_csv(TEST_DIR / "data/test.csv")
    yield data.astype("double")

@pytest.fixture()
def train_csv():
    data = pd.read_csv(TEST_DIR / "data/train.csv")
    yield data.astype("double")