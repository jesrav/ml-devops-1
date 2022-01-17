"""
Module that holds the ml model configs.

Any model config that needs to work with the `src/modelling/train_evaluate.py` module,
must conform to this interface specified in the meta class BaseModelConfig.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from sklego.preprocessing import ColumnSelector, ColumnDropper
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from custom_transformers import AddMeanWithinCategory


class BaseModelConfig(ABC):
    """Base class for ml model config."""

    @staticmethod
    @abstractmethod
    def get_pipeline(self, **params):
        """Returns SKLearn compatible ml pipeline

        input:
            params: Parameters for sklearn compatible pipeline.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_hyper_parameter_to_search():
        """Get a list of hyperparameters to try."""
        pass

    @staticmethod
    @abstractmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Saves any plots that are relevant for the fitted pipeline."""
        pass


class LogregConfig(BaseModelConfig):
    """Model config for ML pipeline using a logistic regression model."""

    @staticmethod
    def get_pipeline(**params):
        """Get logistic regression pipeline

        The pipeline works on a dataframe and selects the features.
        The categorical features are replaced with the churn mean within each
        category.

        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        TARGET = "Churn"
        FEATURES = [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        CAT_COLS = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]

        churn_group_mean_transformer = AddMeanWithinCategory(
            cat_cols=CAT_COLS,
            target_col="Churn",
            new_col_names=[col + "_Churn" for col in CAT_COLS],
        )

        vanilla_pipeline = Pipeline(
            [
                ("select_columns", ColumnSelector(FEATURES + [TARGET])),
                ("create_churn_group_means", churn_group_mean_transformer),
                ("drop_cat_cols_and_target", ColumnDropper(CAT_COLS + [TARGET])),
                ("classifier", LogisticRegression())
            ]
        )
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def get_hyper_parameter_to_search():
        return {'classifier__C': [0.1]}


    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Logreg pipeline does not have any plots for the fitted model."""
        pass


class RandomForestConfig(BaseModelConfig):
    """Model config for ML pipeline using a random forest model."""

    @staticmethod
    def get_pipeline(**params):
        """Get random forest pipeline

        The pipeline works on a dataframe and selects the features.
        The categorical features are replaced with the churn mean within each
        category.

        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        TARGET = "Churn"
        FEATURES = [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        CAT_COLS = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        churn_group_mean_transformer = AddMeanWithinCategory(
            cat_cols=CAT_COLS,
            target_col="Churn",
            new_col_names=[col + "_Churn" for col in CAT_COLS],
        )

        vanilla_pipeline = Pipeline(
            [
                ("select_columns", ColumnSelector(FEATURES + [TARGET])),
                ("create_churn_group_means", churn_group_mean_transformer),
                ("drop_cat_cols_and_target", ColumnDropper(CAT_COLS + [TARGET])),
                ("classifier", RandomForestClassifier())
            ]
        )
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        rf_features = pipeline["classifier"].feature_names_in_
        rf_feature_importances = pipeline["classifier"].feature_importances_
        feature_importance_df = pd.DataFrame(
            zip(
                rf_features,
                rf_feature_importances,
            ),
            columns=["feature", "importance"],
        ).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(20, 20))
        ax = sns.barplot(x="feature", y="importance", data=feature_importance_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)
        plt.xlabel('feature', fontsize=24)
        plt.ylabel('importance', fontsize=24)
        fig = ax.get_figure()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(Path(out_dir) / Path("random_forest_feature_importances.png"))

    @staticmethod
    def get_hyper_parameter_to_search():
        return {'classifier__n_estimators': [100]}
