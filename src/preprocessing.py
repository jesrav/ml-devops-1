"""Preprocess the raw data to prepare for modelling"""
import pandas as pd

from src.common import import_data
from src import config


def add_churn_target(dataf: pd.DataFrame) -> pd.DataFrame:
    """Add target for weather the customer churned."""
    dataf = dataf.copy()
    dataf['Churn'] = dataf['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return dataf


if __name__ == '__main__':
    raw_df = import_data(config.RAW_DATA_PATH)
    processed_df = add_churn_target(raw_df)
    processed_df.to_csv(config.PROCESSED_DATA_PATH)