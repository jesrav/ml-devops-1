"""
Module that contains custom sklearn compatible transformer classes.

Author: Jes Ravnbøl
Created: 2022-01-17
"""
from typing import List, Union

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class AddMeanWithinCategory(TransformerMixin, BaseEstimator):
    """
    Transformer that adds column with the mean of a target col within a category
    found in another column.
    """

    def __init__(
        self,
        cat_cols: List[str],
        target_col: str,
        new_col_names: List[str],
        group_means: Union[None, dict] = None,
    ) -> None:
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.new_col_names = new_col_names
        if group_means is None:
            self.group_means = {cat_col: {} for cat_col in cat_cols}
        else:
            self.group_means = group_means

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transfomer, by calculating category means."""
        for cat_col in self.cat_cols:
            self.group_means[cat_col] = (
                X[self.target_col].groupby(X[cat_col]).mean().to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame):
        """Transfomer dataframe by adding category means."""
        X = X.copy()
        for i, cat_col in enumerate(self.cat_cols):
            X[self.new_col_names[i]] = X[cat_col].map(self.group_means[cat_col])
        return X

    def get_params(self, deep=True):
        return {
            "cat_cols": self.cat_cols,
            "target_col": self.target_col,
            "group_means": self.group_means,
            "new_col_names": self.new_col_names,
        }

    def set_params(self, **kwargs):
        for attr in ["cat_cols", "target_col", "new_col_name", "group_means"]:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
