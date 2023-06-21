import numpy as np
from pandas import DataFrame, Series

from abc import ABC, abstractmethod

import sklearn.metrics as metrics
from scikitplot.helpers import cumulative_gain_curve


class Report:
    def __init__(self, y, y_pred, y_pred_proba, scores, tables, visualizations):
        self._y = y
        self._y_pred = y_pred
        self._scores = scores
        self._tables = tables
        self._y_pred_proba = y_pred_proba
        self._visualizations = visualizations

    class Scores:
        def __init__(self, accuracy: float = 0.0, cohen_kappa: float = 0.0, mcc: float = 0.0, specificity: float = 0.0):
            self.mcc = mcc
            self.accuracy = accuracy
            self.specificity = specificity
            self.cohen_kappa = cohen_kappa

    class Tables:
        def __init__(self, confusion_matrix, classification_report):
            self.confusion_matrix = confusion_matrix
            self.classification_report = classification_report

    class Visualizations:
        def __init__(self, roc, precision_recall, cumulative_gain, lift):
            self.roc = roc
            self.lift = lift
            self.cumulative_gain = cumulative_gain
            self.precision_recall = precision_recall

    @staticmethod
    def create_report(model, data: (DataFrame, Series), is_keras: bool = False):
        x, y = data

        y_pred = model.predict(x)
        if is_keras:
            y_pred_proba = _predict_proba(y_pred)
            y_pred = np.array(model.predict(x) > 0.5, dtype=np.int32)
        else:
            y_pred_proba = model.predict_proba(x)

        y_pred_proba_alt = y_pred_proba[:, 1]
        confusion_matrix = metrics.confusion_matrix(y, y_pred)

        return Report(
            y=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            scores=Report.Scores(
                mcc=metrics.matthews_corrcoef(y, y_pred),
                accuracy=metrics.accuracy_score(y, y_pred),
                cohen_kappa=metrics.cohen_kappa_score(y, y_pred),
                specificity=_calculate_specificity(confusion_matrix)
            ),
            tables=Report.Tables(
                confusion_matrix=confusion_matrix,
                classification_report=metrics.classification_report(y, y_pred, output_dict=True)
            ),
            visualizations=Report.Visualizations(
                roc=_calculate_roc_curve(y, y_pred_proba_alt),
                lift=_calculate_lift_curve(y, y_pred_proba),
                cumulative_gain=_calculate_cumulative_gain_curve(y, y_pred_proba),
                precision_recall=_calculate_precision_recall_curve(y, y_pred_proba_alt),
            )
        )

    # @abstractmethod
    def plot_roc(self):
        pass

    # @abstractmethod
    def plot_lift(self):
        pass

    # @abstractmethod
    def plot_cumulative_gain(self):
        pass

    # @abstractmethod
    def plot_precision_recall(self):
        pass

    # @abstractmethod
    def display_confusion_matrix(self):
        pass

    # @abstractmethod
    def display_classification_report(self):
        pass

    # @abstractmethod
    def display_scores(self):
        pass


def _predict_proba(results):
    new_results = []
    for result in results:
        new_results.append([1 - result[0], result[0]])

    return np.array(new_results)


def _calculate_specificity(matrix):
    tn, fp, fn, tp = matrix.ravel()
    return tn / (tn + fp)


def _calculate_roc_curve(y, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)
    return {'fpr': fpr, 'tpr': tpr, 'auc': metrics.auc(fpr, tpr)}


def _calculate_precision_recall_curve(y, y_pred_proba):
    precision, recall, _ = metrics.precision_recall_curve(y, y_pred_proba)
    return {'precision': precision, 'recall': recall}


def _calculate_cumulative_gain_curve(y, y_pred_proba):
    percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0])
    percentages, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1])

    return percentages, gains1, gains2


def _calculate_lift_curve(y, y_pred_proba):
    classes = np.unique(y)
    percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0], classes[0])
    percentages, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1], classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:] / percentages
    gains2 = gains2[1:] / percentages

    return percentages, gains1, gains2
