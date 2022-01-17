"""
Module for training and evaluating a churn model.

Author: Jes Ravnbøl
Created: 2022-01-17
"""
from pathlib import Path
from typing import Type

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from evaluation import Evaluation
from model_configs import BaseModelConfig, LogregConfig, RandomForestConfig
from plotting import plot_univariate_hist, plot_correlation_heatmap, plot_univariate_dist, plot_bar_chart

TARGET = "Churn"

def import_data(pth: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        return pd.read_csv(pth)
    except FileNotFoundError:
        raise FileNotFoundError(f"No file found in path {pth}.")


def add_churn_target(dataf: pd.DataFrame) -> pd.DataFrame:
    """Add target for weather the customer churned."""
    dataf = dataf.copy()
    dataf["Churn"] = dataf["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return dataf


def perform_eda(dataf: pd.DataFrame, out_dir: str) -> None:
    """
    Perform eda on df and save figures to folder.

    The function saves 3 plots to out_dir.
    - A histogram of Churn
    - A histogram of customer age
    - A feature correlation heatmap

    input:
            df: pandas dataframe
            out_dir: Directory to save plots to.
    output:
            None
    """
    # Set paths
    churn_hist_path = Path(out_dir) / Path("churn_hist.jpg")
    age_hist_path = Path(out_dir) / Path("age_hist.jpg")
    corr_heatmap_path = Path(out_dir) / Path("corr_heatmap.jpg")
    dist_total_transaction_path = Path(out_dir) / Path("dist_total_transaction.jpg")
    barchart_marital_status_path = Path(out_dir) / Path("barchart_marital_status.jpg")

    # Create plots
    plot_univariate_hist(dataf, "Churn", str(churn_hist_path))
    plot_univariate_hist(dataf, "Customer_Age", str(age_hist_path))
    plot_correlation_heatmap(dataf, str(corr_heatmap_path))
    plot_univariate_dist(dataf, "Customer_Age", str(dist_total_transaction_path))
    plot_bar_chart(dataf, "Marital_Status", str(barchart_marital_status_path))


def train_model_cross_validation(
        dataf: pd.DataFrame,
        model_config: Type[BaseModelConfig],
        run_name: str,
        model_artifact_path: str,
        model_evaluation_plots_dir: str
) -> None:
    """
    Train and evaluate ml model pipeline

    input:
        dataf: Dataframe with modelling data.
        model_config: Model configuration. Must follow the interface found in BaseModelConfig.
        run_name: Name of the training run.
        model_artifact_path: Path where the model artifacts is saved.
        model_evaluation_plots_dir: Directory where the model evaluation plots are saved.

    output:
            None
    """
    # Splitting in train and test
    train_df, test_df = train_test_split(
        dataf, test_size=0.3, random_state=42
    )

    # Fit ml pipeline using cross validation and use best model.
    cv_pipeline = GridSearchCV(
        estimator=model_config.get_pipeline(),
        param_grid=model_config.get_hyper_parameter_to_search(),
        cv=5
    )
    cv_pipeline.fit(train_df, train_df[TARGET])
    best_pipeline = cv_pipeline.best_estimator_

    # Save plots specific to the fitted model
    model_config.save_fitted_pipeline_plots(
        pipeline=best_pipeline,
        out_dir=str(model_evaluation_plots_dir)
    )

    # Evaluate ml pipeline on test set
    y_test_probas = best_pipeline.predict_proba(test_df)
    test_evaluation = Evaluation(
        y_true=test_df[TARGET], y_proba=y_test_probas, prediction_threshold=0.5
    )
    test_evaluation.save_evaluation_artifacts(
        outdir=model_evaluation_plots_dir, artifact_prefix=f"{run_name}_test"
    )

    # Evaluate ml pipeline on train set
    y_train_probas = best_pipeline.predict_proba(train_df)
    train_evaluation = Evaluation(
       y_true=train_df[TARGET], y_proba=y_train_probas, prediction_threshold=0.5
    )
    train_evaluation.save_evaluation_artifacts(
        outdir=model_evaluation_plots_dir, artifact_prefix=f"{run_name}_train"
    )

    # Serialize ml pipeline object.
    joblib.dump(best_pipeline, model_artifact_path)


if __name__ == '__main__':
    # Get and process data
    df = import_data("data/bank_data.csv")
    df = add_churn_target(df)

    # Exploratory data analysis
    perform_eda(df, out_dir="images")

    # Train and evaluate logistic regression model
    train_model_cross_validation(
        dataf=df,
        model_config=LogregConfig,
        run_name="logreg",
        model_artifact_path="models/logistic_regression_model.pkl",
        model_evaluation_plots_dir="results"
    )

    # Train and evaluate random forest model
    train_model_cross_validation(
        dataf=df,
        model_config=RandomForestConfig,
        run_name="random_forest",
        model_artifact_path="models/random_forest_model.pkl",
        model_evaluation_plots_dir="results"
    )
