from pathlib import Path
import tempfile

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import add_churn_target
from src.utils import import_data
from custom_transformers import AddMeanWithinCategory
from src.modelling.eda import perform_eda
from src.data.feature_engineering import add_features
from train_evaluate import train_and_evaluate
from model_configs import RandomForestConfig
from src.logger import logger
import config


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
            "Churn": [0, 0, 1],
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


def test_eda(perform_eda_func):
    """
    Test that perform_eda function produces 3 plots
    """
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change the config object to point to temp paths
        config.CHURN_HIST_PATH = f"{tmpdirname}/churn_hist.jpg"
        config.AGE_HIST_PATH = f"{tmpdirname}/age_hist.jpg"
        config.CORR_HEATMAP_PATH = f"{tmpdirname}/corr_heatmap.jpg"

        # Load and preprocess data
        df = import_data("data/bank_data.csv")
        df = add_churn_target(df)

        # Perform eda
        perform_eda_func(df)

        # Check if plots exist.
        churn_hist_exists = Path(config.CHURN_HIST_PATH).is_file()
        age_hist_exists =  Path(config.AGE_HIST_PATH).is_file()
        corr_plot_exists =  Path(config.CORR_HEATMAP_PATH).is_file()
    if churn_hist_exists and age_hist_exists and corr_plot_exists:
        logger.info("Testing perform_eda: SUCCESS")
    else:
        if not churn_hist_exists:
            logger.info("Testing perform_eda: Fails - Churn histogram not created.")
        if not age_hist_exists:
            logger.info("Testing perform_eda: Fails - Age histogram not created.")
        if not corr_plot_exists:
            logger.info("Testing perform_eda: Fails - Correlation heatmap not created.")


def test_train_and_evaluate(train_and_evaluate_func):
    """
    test train_and_evaluate using random forest model config.
    """

    # Get modelling data
    df = import_data("data/bank_data.csv")
    df = add_churn_target(df)
    df = add_features(df)

    logger.info(
        "Running entire training job for random forest model to test train_evaluate function:"
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        train_and_evaluate_func(
            dataf=df,
            model_config=RandomForestConfig,
            run_name="rf",
            artifact_dir=tmpdirname
        )
        expected_file_names = [
            "rf_test_metrics.json",
            "rf_test_auc_plot.png",
            "rf_test_precision_recall_plot.png",
            "rf_test_probability_calibration_plot.png",
            "rf_train_metrics.json",
            "rf_train_auc_plot.png",
            "rf_train_precision_recall_plot.png",
            "rf_train_probability_calibration_plot.png",
            "random_forest_feature_importances.png"
        ]
        expected_file_paths = [Path(tmpdirname) / Path(fn) for fn in expected_file_names]
        expected_file_paths_exists = {
            fn: Path(expected_file_paths[i]).is_file()
            for i, fn in enumerate(expected_file_names)
        }
    if all(expected_file_paths_exists.values()):
        logger.info("Testing train_and_evaluate: SUCCESS")
    else:
        for fn, fp_is_file in expected_file_paths_exists.items():
            if not fp_is_file:
                logger.info(f"Testing train_and_evaluate: Fails - {fn} not created.")


if __name__ == "__main__":
    test_import(import_data)
    test_import_raises_right_error(import_data)
    test_add_mean_within_category(AddMeanWithinCategory)
    test_add_churn_target(add_churn_target)
    test_eda(perform_eda)
    test_train_and_evaluate(train_and_evaluate)
