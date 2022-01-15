import importlib

from sklearn.model_selection import train_test_split
import click

from src.common import import_data
import src.config as config
from src.logger import logger
from src.modelling.evaluation import Evaluation


@click.command()
@click.argument(
    'model-config-module',
    type=str,
)
@click.argument(
    'run-name',
    type=str,
)
def main(model_config_module, run_name):
    logger.info(f"Loading modelling data from {config.MODELLING_DATA_PATH}")
    df = import_data(config.MODELLING_DATA_PATH)

    logger.info(f"Importing model configuration from module {model_config_module}.")
    model_config = importlib.import_module(model_config_module)

    logger.info(f"Splitting modelliung data in test and train set.")
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42
    )

    logger.info(f"Fitting ml pipeline.")
    model_config.pipeline.fit(train_df, train_df[model_config.TARGET])

    logger.info(f"Evaluating ml pipeline on test set and saving artifacts to {config.ARTIFACT_DIR}.")
    y_test_probas = model_config.pipeline.predict_proba(test_df)
    test_evaluation = Evaluation(
        y_true=test_df[model_config.TARGET], y_proba=y_test_probas, prediction_threshold=0.5
    )
    test_evaluation.save_evaluation_artifacts(
        outdir=config.ARTIFACT_DIR, artifact_prefix=f"{run_name}_test"
    )

    logger.info(f"Evaluating ml pipeline on train set and saving artifacts to {config.ARTIFACT_DIR}.")
    y_train_probas = model_config.pipeline.predict_proba(train_df)
    train_evaluation = Evaluation(
       y_true=train_df[model_config.TARGET], y_proba=y_train_probas, prediction_threshold=0.5
    )
    train_evaluation.save_evaluation_artifacts(
        outdir=config.ARTIFACT_DIR, artifact_prefix=f"{run_name}_train"
    )


if __name__ == '__main__':
    main()
