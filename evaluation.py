"""Module that contains functionality to evaluate model performance.
The main function is evaluate, which returns metrics and plots about out of sample predictions.
"""
import json
from pathlib import Path
from typing import Union

import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Evaluation:
    """Class to do evaluation on a modelling_artifacts performance"""

    def __init__(
        self, y_true: np.ndarray, y_proba: np.ndarray, prediction_threshold: float,
    ) -> None:
        """Construct the Evaluation object
        :y_true: y_true (array-like, shape (n_samples)) – Ground truth (correct) target values.
        :y_proba: (array-like, shape (n_samples, 2)) – Prediction probabilities for the two classes
            returned by a classifier.
        :prediction_threshold: Threshold over which to predict a
        :return: None
        """

        if not 0 < prediction_threshold < 1:
            raise ValueError("prediction_threshold needs to be between 0 and 1.")

        if len(y_true) != len(y_proba):
            raise ValueError("Length of y_true and y_proba must be the same.")
        self.y_true = y_true
        self.y_proba = y_proba
        self.prediction_threshold = prediction_threshold
        self.y_pred = self.y_proba[:, 1] > self.prediction_threshold

    def get_classification_report(self) -> dict:
        """Get the classification report as a dictionary."""
        return classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
        )

    def plot_auc(self, outpath: Path) -> None:
        """Plot AUC curve
        The plot is saved to outpath
        :outpath: Outpath for plot.
        :return: None
        """
        plt.subplots()
        skplt.metrics.plot_roc(
            self.y_true,
            self.y_proba,
            plot_micro=False,
            plot_macro=False,
            classes_to_plot=[1],
            title="ROC Curve for SUFFL=1",
        )
        plt.savefig(str(outpath))
        plt.close()

    def plot_probability_calibration_curve(self, outpath: Path) -> None:
        """Plot probability calibration curve"""
        plt.subplots()
        skplt.metrics.plot_calibration_curve(
            self.y_true,
            probas_list=[self.y_proba],
            title="Calibration plot for probability.",
        )
        plt.savefig(str(outpath))
        plt.close()

    def plot_precision_recall(self, outpath: Path) -> None:
        """Plot precision-recalls curve"""
        plt.subplots()
        skplt.metrics.plot_precision_recall(
            self.y_true,
            self.y_proba,
            plot_micro=False,
            classes_to_plot=[1],
            title="Precision-recall Curve.",
        )
        plt.savefig(str(outpath))
        plt.close()

    def save_evaluation_artifacts(self, outdir: Union[Path, str], artifact_prefix: str) -> None:
        """Save all evaluation artifacts to a folder"""
        self.plot_auc(Path(outdir) / Path(f"{artifact_prefix}_auc_plot.png"))
        self.plot_precision_recall(
            Path(outdir) / Path(f"{artifact_prefix}_precision_recall_plot.png")
        )
        self.plot_probability_calibration_curve(
            Path(outdir) / Path(f"{artifact_prefix}_probability_calibration_plot.png")
        )
        with open(
                Path(outdir) / Path(f"{artifact_prefix}_metrics.json"), "w", encoding="utf8"
        ) as file_handler:
            json.dump(self.get_classification_report(), file_handler)
