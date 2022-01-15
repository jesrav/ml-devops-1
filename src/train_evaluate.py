import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.common import import_data
from src.evaluation import Evaluation
import src.config as config

df = import_data(config.MODELLING_DATA_PATH)


features = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']



def get_feature_names(ml_pipeline):
    ml_pipeline["feature_preprocess"].transformer_list[0][1][2].feature_names_in_ = CATEGORICAL_COLS
    return (
        ml_pipeline["feature_preprocess"].transformer_list[0][1][2].get_feature_names_out().tolist()
        + NUMERICAL_COLS
    )


def get_feature_importances(ml_pipeline):
    return pd.DataFrame(
        zip(get_feature_names(ml_pipeline), ml_pipeline["classifier"].feature_importances_),
        columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=False)
X = df[keep_cols]
y = df["Churn"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)


# grid search
#rfc = RandomForestClassifier(random_state=42)
lrc = LogisticRegression()

# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth' : [4,5,100],
#     'criterion' :['gini', 'entropy']
# }

#cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
#cv_rfc.fit(X_train, y_train)

lrc.fit(X_train, y_train)

#y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
#y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

y_train_preds = lrc.predict(X_train)
y_test_preds = lrc.predict(X_test)
y_train_probas = lrc.predict_proba(X_train)
y_test_probas = lrc.predict_proba(X_test)

test_evaluation = Evaluation(
    y_true=y_test,
    y_proba=y_test_probas,
    prediction_threshold=0.5
)
train_evaluation = Evaluation(
    y_true=y_train,
    y_proba=y_train_probas,
    prediction_threshold=0.5
)
test_evaluation.save_evaluation_artifacts("modelling_artifacts")
