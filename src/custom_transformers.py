"""Module that contains custom sklearn compatible transformer classes."""
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class AddMeanWithinCategory(TransformerMixin, BaseEstimator):
    """
    Transformer that adds column with the mean of a target col within a category found in another column.
    """

    def __init__(self, cat_col: str, target_col: str, new_col_name: str) -> None:
        self.cat_col = cat_col
        self.target_col = target_col
        self.new_col_name = new_col_name
        self.group_means = None

    def fit(self, X: pd.DataFrame, y=None):
        self.group_means = X[self.target_col].groupby(X[self.cat_col]).mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X[self.new_col_name] = X[self.cat_col].map(self.group_means)
        return X

    def get_params(self, deep=True):
        return {
            "cat_col": self.cat_col,
            "target_col": self.target_col,
            "group_means": self.group_means,
            "new_col_name": self.new_col_name,
        }

    def set_params(self, **kwargs):
        for attr in ["cat_col", "target_col", "new_col_name", "group_means"]:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
