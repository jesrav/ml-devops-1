"""Model config for ML pipeline using a logistig regression model."""
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklego.preprocessing import ColumnSelector, ColumnDropper
import pandas as pd
import seaborn as sns

from src.modelling.custom_transformers import AddMeanWithinCategory

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
CAT_COLS=[
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

pipeline = Pipeline(
    [
        ("select_columns", ColumnSelector(FEATURES + [TARGET])),
        ("create_churn_group_means", churn_group_mean_transformer),
        ("drop_cat_cols_and_target", ColumnDropper(CAT_COLS + [TARGET])),
        ("classifier", RandomForestClassifier())
    ]
)

def save_fitted_pipeline_plots(out_dir: str):
    rf_features = pipeline["classifier"].feature_names_in_
    rf_feature_importances = pipeline["classifier"].feature_importances_
    feature_importance_df = pd.DataFrame(
            zip(
                rf_features,
                rf_feature_importances,
            ),
            columns=["feature", "importance"],
        ).sort_values(by="importance", ascending=False)

    barplot_fig = sns.barplot(x="feature", y="importance", data=feature_importance_df).get_figure()
    barplot_fig.savefig(Path(out_dir) / Path("random_forest_feature_importances.png"))
