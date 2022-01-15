from src.logger import logger
from src.common import import_data
from src.data.feature_engineering import get_mean_within_category

logging.basicConfig(
    filename='../logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data_func):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data_func("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """


def test_get_mean_within_category(encoder_helper):
    """
    test encoder helper
    """
    get_mean_within_category$


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """


def test_train_models(train_models):
    """
    test train_models
    """


if __name__ == "__main__":
    test_import(import_data)
