"""
Module for plotting functionality.

Author: Jes Ravnbøl
Created: 2022-01-17
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def plot_univariate_hist(dataf: pd.DataFrame, col_name: str, out_path: str) -> None:
    """
    Plot histogram.

    input:
        dataf: pandas dataframe with preprocessed data. Must include chrun column.
        col_name: Name of column to make a histogram of.
        out_path: String with outpath for plot.

    output:
            None
            :rtype: object
    """
    plt.figure(figsize=(20, 10))
    plt.xlabel(col_name)
    fig = dataf[col_name].hist().get_figure()
    fig.savefig(out_path)


def plot_univariate_dist(dataf: pd.DataFrame, col_name: str, out_path: str) -> None:
    """
    Plot distribution.

    input:
        dataf: pandas dataframe with preprocessed data. Must include chrun column.
        col_name: Name of column to make a histogram of.
        out_path: String with outpath for plot.

    output:
            None
            :rtype: object
    """
    plt.figure(figsize=(20, 10))
    plt.xlabel(col_name)
    plt.figure(figsize=(20, 10))
    dist_fit = sns.distplot(dataf[col_name]).get_figure()
    dist_fit.savefig(out_path)


def plot_bar_chart(dataf: pd.DataFrame, col_name: str, out_path: str) -> None:
    """
    Plot bar chart.

    input:
        dataf: pandas dataframe with preprocessed data. Must include chrun column.
        col_name: Name of column to make a histogram of.
        out_path: String with outpath for plot.

    output:
            None
            :rtype: object
    """
    plt.figure(figsize=(20, 10))
    plt.xlabel(col_name)
    fig = dataf[col_name].value_counts('normalize').plot(kind='bar').get_figure()
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
