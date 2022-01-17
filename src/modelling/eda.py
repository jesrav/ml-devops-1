"""module for for explorative data analysis."""
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src import config
from src.utils import import_data
from src.logger import logger

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def plot_univariate_hist(dataf: pd.DataFrame, col_name: str, out_path: str) -> None:
    """
    Plot histogram of churn.

    input:
        dataf: pandas dataframe with preprocessed data. Must include chrun column.
        out_path: String with outpath for plot.

    output:
            None
    """
    plt.figure(figsize=(20, 10))
    plt.xlabel(col_name)
    fig = dataf[col_name].hist().get_figure()
    fig.savefig(out_path)


def plot_correlation_heatmap(dataf: pd.DataFrame, out_path: str) -> None:
    """
    Plot histogram of churn.

    input:
        dataf: pandas dataframe with preprocessed data. Must include churn column.
        out_path: String with outpath for plot.

    output:
            None
    """
    plt.figure(figsize=(20, 10))
    corr_heatmap_fig = sns.heatmap(
        dataf.corr(), annot=False, cmap="Dark2_r", linewidths=2
    ).get_figure()
    corr_heatmap_fig.savefig(out_path)


def perform_eda(dataf: pd.DataFrame) -> None:
    plot_univariate_hist(dataf, "Churn", config.CHURN_HIST_PATH)
    plot_univariate_hist(dataf, "Customer_Age", config.AGE_HIST_PATH)
    plot_correlation_heatmap(dataf, config.CORR_HEATMAP_PATH)


if __name__ == "__main__":
    logger.info(f"Import preprocessed data from {config.PROCESSED_DATA_PATH}")
    df = import_data(config.PROCESSED_DATA_PATH)

    logger.info(f"Perform eda.")
    perform_eda(df)