from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.modelling.custom_transformers import AddMeanWithinCategory

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
]

columns_selector = ColumnTransformer(
    transformers=[("selector", "passthrough", FEATURES)],
    remainder="drop",
)

churn_group_mean_transformer = AddMeanWithinCategory(
    cat_cols=[
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ],
    target_col="Churn",
    new_col_names=[
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ],
)


pipeline = Pipeline(
    [
        ("column_selector", columns_selector),
        ("column_selector", churn_group_mean_transformer),
        ("classifier", LogisticRegression())
    ]
)
