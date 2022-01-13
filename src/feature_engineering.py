"""Module for feature engineering functionality"""
import pandas as pd
import numpy as np

from src import config


def add_mean_within_category(
        df: pd.DataFrame, cat_col: str, target_col: str, new_col_name: str
) -> pd.DataFrame:
    """
    Add column with the mean of `target_col`withing a category found in cat_col.
    input:
            df: Dataframe to add category mean to
            cat_col: Name of column with the categories to perform the grouped means within.
            target_col: Name of column with the values that we that the mean of.
            new_col_name: Name of the new result column

    output:
            Dataframe with the added column `new_col_name`
    """
    df = df.copy()
    cat_lst = []
    cat_groups = df.groupby(cat_col).mean()[target_col]
    for val in df[cat_col]:
        cat_lst.append(cat_groups.loc[val])
    df[new_col_name] = cat_lst
    return df


def get_mean_within_category(
        df: pd.DataFrame, cat_col: str, target_col: str
) -> pd.Series:
    """
    Add column with the mean of `target_col`withing a category found in cat_col.
    input:
            df: Dataframe to add category mean to
            cat_col: Name of column with the categories to perform the grouped means within.
            target_col: Name of column with the values that we that the mean of.
            new_col_name: Name of the new result column

    output:
            Dataframe with the added column `new_col_name`
    """
    df = df.copy()
    return df[target_col].groupby(df[cat_col]).transform(np.mean)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features"""

    category_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
    for col in category_cols:
        df[col + "_Churn"] = get_mean_within_category(df, col, "Churn")
    return df


if __name__ == '__main__':
    print("Read processed data.")
    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    print("Add features.")
    df = add_features(df)

    print(f"Write data with features to {config.MODELLING_DATA_PATH}.")
    df.to_csv(config.MODELLING_DATA_PATH)
