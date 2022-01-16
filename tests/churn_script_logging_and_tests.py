#import  logging

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import add_churn_target
from src.utils import import_data
from src.modelling.custom_transformers import AddMeanWithinCategory
from src.logger import logger


# logging.basicConfig(
#     filename='logs/churn_library.log',
#     level=logging.INFO,
#     filemode='w',
#     format='%(name)s - %(levelname)s - %(message)s')



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


def test_import_raises_right_error(import_data_func):
    """test data import throws the correct error"""
    expected_error = "No file found in path /wrong/path/to/a.csv."
    try:
        with pytest.raises(FileNotFoundError) as error_msg:
            import_data_func("/wrong/path/to/a.csv")
        assert str(error_msg.value) == expected_error, \
        f"'{str(error_msg.value)}' is not equal expected error '{expected_error}'"

        logger.info("Testing import_data raises the right error: SUCCESS")
    except AssertionError as e:
        logger.info(
            f"Testing that import_data raises the right error: FAILS -"
            f"import_data does not raise the right error when wrong path is supplied: {e}"
        )


def test_add_mean_within_category(transformer_cls):
    group_mean_transformer = transformer_cls(
        cat_cols=["group"], target_col="value", new_col_names=["grouped_mean"]
    )
    input_df = pd.DataFrame(
        {
            "group": ["a", "b", "a", "b", "c"],
            "value": [1, 3, 3, 2, 4],
        }
    )
    expected_result = np.array([2, 2.5, 2, 2.5, 4])
    try:
        assert (
            group_mean_transformer.fit_transform(input_df)["grouped_mean"].values == expected_result
        ).all()
        logger.info("Testing AddMeanWithinCategory transformer: SUCCESS")
    except AssertionError as e:
        logger.info(
            f"Testing AddMeanWithinCategory transformer: FAILS - "
            f"The transformer does not produce the expected results : {e}."
        )


def test_add_churn_target(add_churn_target_func):

    input_df = pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Existing Customer",
                "Attrited Customer",
            ],
            "dummy": [1, 3, 3],
        }
    )
    expected_result = pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Existing Customer",
                "Attrited Customer",
            ],
            "dummy": [1, 3, 3],
            "Churn": [0, 0, 0],
        }
    )
    try:
        pd.testing.assert_frame_equal(add_churn_target_func(input_df), expected_result)
        logger.info("Testing add_churn_target: SUCCESS")
    except AssertionError as e:
        logger.info(
            f"Testing add_churn_target: FAILS -"
            f"The transformed data frame does not match the expected results: {e}"
        )

# def test_eda(perform_eda):
#     """
#     test perform eda function
#     """
#
#
# def test_get_mean_within_category(encoder_helper):
#     """
#     test encoder helper
#     """
#     get_mean_within_category$
#
#
# def test_perform_feature_engineering(perform_feature_engineering):
#     """
#     test perform_feature_engineering
#     """
#
#
# def test_train_models(train_models):
#     """
#     test train_models
#     """


if __name__ == "__main__":
    test_import(import_data)
    test_import_raises_right_error(import_data)
    test_add_mean_within_category(AddMeanWithinCategory)
    test_add_churn_target(add_churn_target)
