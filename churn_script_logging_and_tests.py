"""
Script for testing the functions in the churn project.

Author: Jes Ravnbøl
Created: 2022-01-17
"""
import logging
from pathlib import Path
import tempfile

import pytest
import pandas as pd
import numpy as np

from churn_library import (
    add_churn_target, import_data, perform_eda, train_model_cross_validation
)
from custom_transformers import AddMeanWithinCategory
from model_configs import RandomForestConfig

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data_func):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data_func("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_import_raises_right_error(import_data_func):
    """test data import throws the correct error"""
    expected_error = "No file found in path /wrong/path/to/a.csv."
    try:
        with pytest.raises(FileNotFoundError) as error_msg:
            import_data_func("/wrong/path/to/a.csv")
        assert str(error_msg.value) == expected_error, \
        f"'{str(error_msg.value)}' is not equal expected error '{expected_error}'"

        logging.info("Testing import_data raises the right error: SUCCESS")
    except AssertionError as e:
        logging.info(
            f"Testing that import_data raises the right error: FAILS -"
            f"import_data does not raise the right error when wrong path is supplied: {e}"
        )


def test_add_mean_within_category(transformer_cls):
    """
    Test that the transformer class `AddMeanWithinCategory` works as expected.
    """
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
        logging.info("Testing AddMeanWithinCategory transformer: SUCCESS")
    except AssertionError as e:
        logging.info(
            f"Testing AddMeanWithinCategory transformer: FAILS - "
            f"The transformer does not produce the expected results : {e}."
        )


def test_add_churn_target(add_churn_target_func):
    """Test that add_churn_target works as expected."""
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
        logging.info("Testing add_churn_target: SUCCESS")
    except AssertionError as e:
        logging.info(
            f"Testing add_churn_target: FAILS -"
            f"The transformed data frame does not match the expected results: {e}"
        )


def test_perform_eda(perform_eda_func):
    """
    Test that perform_eda function produces 3 plots
    """
    # Load and preprocess data
    df = import_data("data/bank_data.csv")
    df = add_churn_target(df)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Perform eda
        perform_eda_func(df, out_dir=tmpdirname)

        expected_churn_hist_path = Path(tmpdirname) / Path("churn_hist.jpg")
        expected_age_hist_path = Path(tmpdirname) / Path("age_hist.jpg")
        expected_corr_heatmap_path = Path(tmpdirname) / Path("corr_heatmap.jpg")
        expected_dist_total_transaction_path = Path(tmpdirname) / Path("dist_total_transaction.jpg")
        expected_barchart_marital_status_path = Path(tmpdirname) / Path("barchart_marital_status.jpg")

        # Check if expected plots have been created.
        churn_hist_exists = expected_churn_hist_path.is_file()
        age_hist_exists =  expected_age_hist_path.is_file()
        corr_plot_exists =  expected_corr_heatmap_path.is_file()
        dist_total_transaction_plot_exists = expected_dist_total_transaction_path.is_file()
        barchart_marital_status_plot_exists = expected_barchart_marital_status_path.is_file()


    if all([
        churn_hist_exists,
        age_hist_exists,
        corr_plot_exists,
        dist_total_transaction_plot_exists,
        barchart_marital_status_plot_exists,
    ]):
        logging.info("Testing perform_eda: SUCCESS")
    else:
        if not churn_hist_exists:
            logging.info("Testing perform_eda: Fails - Churn histogram not created.")
        if not age_hist_exists:
            logging.info("Testing perform_eda: Fails - Age histogram not created.")
        if not corr_plot_exists:
            logging.info("Testing perform_eda: Fails - Correlation heatmap not created.")
        if not dist_total_transaction_plot_exists:
            logging.info(
                "Testing perform_eda: Fails - Distribution plot for total transactions not created."
            )
        if not barchart_marital_status_plot_exists:
            logging.info(
                "Testing perform_eda: Fails - Bar chart for marital status not created."
            )


def test_train_model_cross_validation(train_model_cross_validation_func):
    """
    Test train_and_evaluate using random forest model config.
    We test that the relevant artifacts from the function are created.
    """

    # Get modelling data
    df = import_data("data/bank_data.csv")
    df = add_churn_target(df)

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        expected_model_artifact_path = Path(tmpdir1) / Path("model.pkl")
        expected_evaluation_file_names = [
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
        expected_evaluation_file_paths = [
            Path(tmpdir2) / Path(fn) for fn in expected_evaluation_file_names
        ]

        train_model_cross_validation(
            dataf=df,
            model_config=RandomForestConfig,
            run_name="rf",
            model_artifact_path=str(expected_model_artifact_path),
            model_evaluation_plots_dir=tmpdir2,
        )

        expected_evaluation_file_paths_exists = {
            fn: Path(expected_evaluation_file_paths[i]).is_file()
            for i, fn in enumerate(expected_evaluation_file_names)
        }
        expected_model_artifact_path_exists = expected_model_artifact_path.is_file()

    if all(expected_evaluation_file_paths_exists.values()) and expected_model_artifact_path_exists:
        logging.info("Testing train_and_evaluate: SUCCESS")
    else:
        if not expected_model_artifact_path_exists:
            logging.info(f"Testing train_and_evaluate: Fails - {expected_model_artifact_path} not created.")
        for fn, fp_is_file in expected_evaluation_file_paths_exists.items():
            if not fp_is_file:
                logging.info(f"Testing train_and_evaluate: Fails - {fn} not created.")


if __name__ == "__main__":
    test_import(import_data)
    test_import_raises_right_error(import_data)
    test_add_mean_within_category(AddMeanWithinCategory)
    test_add_churn_target(add_churn_target)
    test_perform_eda(perform_eda)
    test_train_model_cross_validation(train_model_cross_validation)
