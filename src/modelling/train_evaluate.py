from sklearn.model_selection import train_test_split

from src.common import import_data
import src.config as config
from src.modelling.evaluation import Evaluation
from src.modelling.model_configs import logreg as model_conf


def main()
    df = import_data(config.PROCESSED_DATA_PATH)

    # train test split
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42
    )

    model_conf.pipeline.fit(train_df, train_df[model_conf.TARGET])

    y_train_preds = model_conf.pipeline.predict(train_df)
    y_test_preds = model_conf.pipeline.predict(test_df)
    y_train_probas = model_conf.pipeline.predict_proba(train_df)
    y_test_probas = model_conf.pipeline.predict_proba(test_df)

    test_evaluation = Evaluation(
        y_true=test_df[model_conf.TARGET], y_proba=y_test_probas, prediction_threshold=0.5
    )
    train_evaluation = Evaluation(
       y_true=train_df[model_conf.TARGET], y_proba=y_train_probas, prediction_threshold=0.5
    )
    test_evaluation.save_evaluation_artifacts("modelling_artifacts", artifact_prefix="logreg_train")
    test_evaluation.save_evaluation_artifacts("modelling_artifacts", "logreg_train")
