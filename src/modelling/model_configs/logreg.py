"""Model config for ML pipeline using a logistig regression model."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklego.preprocessing import ColumnSelector, ColumnDropper
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
        ("classifier", LogisticRegression())
    ]
)


#
# def get_feature_names(ml_pipeline):
#     ml_pipeline["feature_preprocess"].transformer_list[0][1][
#         2
#     ].feature_names_in_ = CATEGORICAL_COLS
#     return (
#         ml_pipeline["feature_preprocess"]
#         .transformer_list[0][1][2]
#         .get_feature_names_out()
#         .tolist()
#         + NUMERICAL_COLS
#     )
#
#
# def get_feature_importances(ml_pipeline):
#     return pd.DataFrame(
#         zip(
#             get_feature_names(ml_pipeline),
#             ml_pipeline["classifier"].feature_importances_,
#         ),
#         columns=["feature", "importance"],
#     ).sort_values(by="importance", ascending=False)
