import pandas as pd


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        return pd.read_csv(pth)
    except FileNotFoundError:
        raise FileNotFoundError(f"No file found in path {pth}.")
