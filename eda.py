import pandas as pd
from matplotlib import pyplot as plt

import config
from common import import_data


def plot_churn_hist(dataf: pd.DataFrame, out_path: str) -> None:
    """
    Plot histogram of churn.

    input:
        dataf: pandas dataframe with preprocessed data. Must include chrun column.
        out_path: String with outpath for plot.

    output:
            None
    """
    plt.figure(figsize=(20, 10))
    fig = dataf['Churn'].hist().get_figure()
    fig.savefig(out_path)


if __name__ == '__main__':
    df = import_data(config.PROCESSED_DATA_PATH)
    plot_churn_hist(df, config.CHURN_HIST_PATH)