import logging

import pytest

from src.utils import import_data

logger = logging.getLogger(__name__)


def test_import():
    """test data import"""
    df = import_data("./data/bank_data.csv")
    logger.info("Testing import_data: SUCCESS")
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_import_raises_right_error():
    """test data import throws the correct error"""
    with pytest.raises(FileNotFoundError) as e:
        import_data("/wrong/path/to/a.csv")
    assert str(e.value) == "No file found in path /wrong/path/to/a.csv."
