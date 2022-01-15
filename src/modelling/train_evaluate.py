import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.common import import_data
from src.modelling.evaluation import Evaluation
import src.config as config
from src.modelling.model_configs.logreg import pipeline

df = import_data(config.MODELLING_DATA_PATH)



def get_feature_names(ml_pipeline):
    ml_pipeline["feature_preprocess"].transformer_list[0][1][
        2
    ].feature_names_in_ = CATEGORICAL_COLS
    return (
        ml_pipeline["feature_preprocess"]
        .transformer_list[0][1][2]
        .get_feature_names_out()
        .tolist()
        + NUMERICAL_COLS
    )


def get_feature_importances(ml_pipeline):
    return pd.DataFrame(
        zip(
            get_feature_names(ml_pipeline),
            ml_pipeline["classifier"].feature_importances_,
        ),
        columns=["feature", "importance"],
    ).sort_values(by="importance", ascending=False)


X = df[keep_cols]
y = df["Churn"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# grid search
# rfc = RandomForestClassifier(random_state=42)
lrc = LogisticRegression()

# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth' : [4,5,100],
#     'criterion' :['gini', 'entropy']
# }

# cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# cv_rfc.fit(X_train, y_train)

lrc.fit(X_train, y_train)

# y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
# y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

y_train_preds = lrc.predict(X_train)
y_test_preds = lrc.predict(X_test)
y_train_probas = lrc.predict_proba(X_train)
y_test_probas = lrc.predict_proba(X_test)

test_evaluation = Evaluation(
    y_true=y_test, y_proba=y_test_probas, prediction_threshold=0.5
)
train_evaluation = Evaluation(
    y_true=y_train, y_proba=y_train_probas, prediction_threshold=0.5
)
test_evaluation.save_evaluation_artifacts("modelling_artifacts")
