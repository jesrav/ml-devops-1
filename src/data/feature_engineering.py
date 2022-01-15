"""Module for feature engineering functionality"""
import pandas as pd

from src import config
from src.common import import_data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features"""
    return df


if __name__ == "__main__":
    print("Read processed data.")
    df = import_data(config.PROCESSED_DATA_PATH)

    print("Add features.")
    df = add_features(df)

    print(f"Write data with features to {config.MODELLING_DATA_PATH}.")
    df.to_csv(config.MODELLING_DATA_PATH)
