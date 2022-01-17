"""Module for training and evaluating a sklearn compatible ML pipeline."""
import importlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import click

from src.utils import import_data
from src import config
from src.logger import logger
from src.modelling.evaluation import Evaluation
from src.modelling.model_configs import BaseModelConfig


MODEL_CONFIGS_MODULE_STR = "src.modelling.model_configs"


def train_and_evaluate(
        dataf: pd.DataFrame,
        model_config: BaseModelConfig,
        run_name: str,
        artifact_dir: str,
        logging: bool = True,
) -> None:
    """
    Train and evaluate ml model pipeline

    input:
        dataf: Dataframe with modelling data.
        model_config: Model configuration
        run_name: Name of the training run.
        artifact_dir: Directory where the modelling artifacts are saved.
        logging: Weather to log

    output:
            None
    """
    logger.info("Creating artifact directory %s if it does not exist", artifact_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=False, exist_ok=True)

    logger.info("Splitting modelling data in test and train set.")
    train_df, test_df = train_test_split(
        dataf, test_size=0.3, random_state=42
    )

    logger.info("Fitting ml pipeline.")
    pipeline = model_config.get_pipeline()
    pipeline.fit(train_df, train_df[config.TARGET])

    logger.info(f"Saving plots specific to the fitted model to {artifact_dir}.")
    model_config.save_fitted_pipeline_plots(pipeline, str(artifact_dir))

    logger.info(
        "Evaluating ml pipeline on test set and saving artifacts to %s ", artifact_dir
    )
    y_test_probas = pipeline.predict_proba(test_df)
    test_evaluation = Evaluation(
        y_true=test_df[config.TARGET], y_proba=y_test_probas, prediction_threshold=0.5
    )
    test_evaluation.save_evaluation_artifacts(
        outdir=artifact_dir, artifact_prefix=f"{run_name}_test"
    )

    logger.info(
        "Evaluating ml pipeline on train set and saving artifacts to %s ", artifact_dir
    )
    y_train_probas = pipeline.predict_proba(train_df)
    train_evaluation = Evaluation(
       y_true=train_df[config.TARGET], y_proba=y_train_probas, prediction_threshold=0.5
    )
    train_evaluation.save_evaluation_artifacts(
        outdir=artifact_dir, artifact_prefix=f"{run_name}_train"
    )

    model_outpath = artifact_dir / Path("model.pkl")
    logger.info(
        "Serialize ml pipeline object to %s", model_outpath
    )
    joblib.dump(pipeline, model_outpath)


@click.command()
@click.argument(
    'model_config_class_name',
    type=str,
)
@click.argument(
    'run-name',
    type=str,
)
@click.argument(
    'artifact-dir',
    type=str,
)
def main(
    model_config_class_name: str,
    run_name: str,
    artifact_dir: str
):
    logger.info("Loading modelling data from %s ", config.MODELLING_DATA_PATH)
    df = import_data(config.MODELLING_DATA_PATH)

    logger.info("Importing model configuration from class from %s ", )
    model_config_module = importlib.import_module(MODEL_CONFIGS_MODULE_STR)
    model_config = getattr(model_config_module, model_config_class_name)

    train_and_evaluate(df, model_config, run_name, artifact_dir)
    

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
