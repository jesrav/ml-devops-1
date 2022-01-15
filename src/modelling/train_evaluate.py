"""Module for training and evaluating a sklearn compatible ML pipeline."""
import importlib

from sklearn.model_selection import train_test_split
import click

from src.utils import import_data
from src import config
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
def train_and_evaluate(model_config_module: str, run_name: str) -> None:
    """
    Train and evaluate ml model pipeline

    input:
        model_config_module: String with module with model config.
            Fx 'src.model.model_configs.my_model_config".
            Module should contain a pipeline object, with a sklearn compatible pipline.
        run_name: Name of the training run.

    output:
            None
    """
    logger.info("Loading modelling data from %d ", config.MODELLING_DATA_PATH)
    df = import_data(config.MODELLING_DATA_PATH)

    logger.info("Importing model configuration from module %d ", model_config_module)
    model_config = importlib.import_module(model_config_module)

    logger.info("Splitting modelliung data in test and train set.")
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42
    )

    logger.info("Fitting ml pipeline.")
    model_config.pipeline.fit(train_df, train_df[model_config.TARGET])

    logger.info(
        "Evaluating ml pipeline on test set and saving artifacts to %d ", config.ARTIFACT_DIR
    )
    y_test_probas = model_config.pipeline.predict_proba(test_df)
    test_evaluation = Evaluation(
        y_true=test_df[model_config.TARGET], y_proba=y_test_probas, prediction_threshold=0.5
    )
    test_evaluation.save_evaluation_artifacts(
        outdir=config.ARTIFACT_DIR, artifact_prefix=f"{run_name}_test"
    )

    logger.info(
        "Evaluating ml pipeline on train set and saving artifacts to %d ", config.ARTIFACT_DIR
    )
    y_train_probas = model_config.pipeline.predict_proba(train_df)
    train_evaluation = Evaluation(
       y_true=train_df[model_config.TARGET], y_proba=y_train_probas, prediction_threshold=0.5
    )
    train_evaluation.save_evaluation_artifacts(
        outdir=config.ARTIFACT_DIR, artifact_prefix=f"{run_name}_train"
    )


if __name__ == '__main__':
    train_and_evaluate() # pylint: disable=no-value-for-parameter
