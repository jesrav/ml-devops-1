"""Preprocess the raw data."""
import pandas as pd

from src.utils import import_data
from src import config
from src.logger import logger


def add_churn_target(dataf: pd.DataFrame) -> pd.DataFrame:
    """Add target for weather the customer churned."""
    dataf = dataf.copy()
    dataf["Churn"] = dataf["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return dataf


if __name__ == "__main__":
    logger.info(f"Loading raw data from {config.RAW_DATA_PATH}")
    raw_df = import_data(config.RAW_DATA_PATH)

    logger.info(f"Preprocessing raw data")
    processed_df = add_churn_target(raw_df)

    logger.info(f"Writing preprocessed data to {config.PROCESSED_DATA_PATH}")
    processed_df.to_csv(config.PROCESSED_DATA_PATH)
