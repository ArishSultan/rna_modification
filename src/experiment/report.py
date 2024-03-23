import numpy as np
from pandas import DataFrame, Series
# import scikitplot

import sklearn.metrics as metrics
# from scikitplot.helpers import cumulative_gain_curve


class Report:
    def __init__(self, scores, tables, visualizations):
        self.scores = scores
        self.tables = tables
        self.visualizations = visualizations

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

    def to_json(self):
        return {
            'scores': {
                'mcc': self.scores.mcc,
                'accuracy': self.scores.accuracy,
                'cohen_kappa': self.scores.cohen_kappa,
                'specificity': self.scores.specificity,
            },
            'tables': {
                'confusion_matrix': self.tables.confusion_matrix,
                'classification_report': self.tables.classification_report,
            },
            'visualizations': {
                'roc': self.visualizations.roc,
                'lift': self.visualizations.lift,
                'cumulative_gain': self.visualizations.cumulative_gain,
                'precision_recall': self.visualizations.precision_recall,
            }
        }

    @staticmethod
    def from_json(jsonobj):
        scores = jsonobj['scores']
        tables = jsonobj['tables']
        visualizations = jsonobj['visualizations']

        return Report(
            scores=Report.Scores(
                mcc=scores['mcc'],
                accuracy=scores['accuracy'],
                cohen_kappa=scores['cohen_kappa'],
                specificity=scores['specificity'],
            ),
            tables=Report.Tables(
                confusion_matrix=tables['confusion_matrix'],
                classification_report=tables['classification_report']
            ),
            visualizations=Report.Visualizations(
                roc=visualizations['roc'],
                lift=visualizations['lift'],
                cumulative_gain=visualizations['cumulative_gain'],
                precision_recall=visualizations['precision_recall'],
            )
        )


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
    # percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0])
    # percentages, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1])

    return {
        'gains1': [],
        'gains2': [],
        'percentages': [],
    }


def _calculate_lift_curve(y, y_pred_proba):
    classes = np.unique(y)
    # percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0], classes[0])
    # percentages, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1], classes[1])

    # percentages = percentages[1:]
    # gains1 = gains1[1:] / percentages
    # gains2 = gains2[1:] / percentages

    return {
        'gains1': [],
        'gains2': [],
        'percentages': [],
    }
