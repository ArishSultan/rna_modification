import numpy as np

from sklearn import metrics
from pandas import DataFrame, Series
from typing import List, Tuple, Dict, Any
from scikitplot.metrics import cumulative_gain_curve


class Report:
    class Scores:
        def __init__(self, accuracy: float = 0.0, cohen_kappa: float = 0.0, mcc: float = 0.0, specificity: float = 0.0):
            self.accuracy = accuracy
            self.cohen_kappa = cohen_kappa
            self.mcc = mcc
            self.specificity = specificity

    class Tables:
        def __init__(self, confusion_matrix: np.ndarray, classification_report: Dict[str, Any]):
            self.confusion_matrix = confusion_matrix
            self.classification_report = classification_report

    class Visualizations:
        def __init__(self, roc: Dict[str, np.ndarray], precision_recall: Dict[str, np.ndarray],
                     cumulative_gain: Dict[str, np.ndarray], lift: Dict[str, np.ndarray]):
            self.roc = roc
            self.precision_recall = precision_recall
            self.cumulative_gain = cumulative_gain
            self.lift = lift

    def __init__(self, scores: Scores, tables: Tables, visualizations: Visualizations):
        self.scores = scores
        self.tables = tables
        self.visualizations = visualizations

    @staticmethod
    def create_report(model: Any, data: Tuple[DataFrame, Series], is_keras: bool = False) -> 'Report':
        x, y = data
        y_pred = model.predict(x)

        if is_keras:
            y_pred_proba = Report._predict_proba(y_pred)
            y_pred = np.array(y_pred > 0.5, dtype=np.int32)
        else:
            y_pred_proba = model.predict_proba(x)

        y_pred_proba_alt = y_pred_proba[:, 1]
        confusion_matrix = metrics.confusion_matrix(y, y_pred)

        return Report(
            scores=Report.Scores(
                accuracy=metrics.accuracy_score(y, y_pred),
                cohen_kappa=metrics.cohen_kappa_score(y, y_pred),
                mcc=metrics.matthews_corrcoef(y, y_pred),
                specificity=Report._calculate_specificity(confusion_matrix)
            ),
            tables=Report.Tables(
                confusion_matrix=confusion_matrix,
                classification_report=metrics.classification_report(y, y_pred, output_dict=True)
            ),
            visualizations=Report.Visualizations(
                roc=Report._calculate_roc_curve(y.values, y_pred_proba_alt),
                precision_recall=Report._calculate_precision_recall_curve(y.values, y_pred_proba_alt),
                cumulative_gain=Report._calculate_cumulative_gain_curve(y.values, y_pred_proba),
                lift=Report._calculate_lift_curve(y.values, y_pred_proba),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scores': vars(self.scores),
            'tabels': {
                'confusion_matrix': self.tables.confusion_matrix.tolist(),
                'classification_report': self.tables.classification_report,
            },
            'visualizations': {
                'roc': self.visualizations.roc,
                'precision_recall': self.visualizations.precision_recall,
                'cumulative_gain': self.visualizations.cumulative_gain,
                'lift': self.visualizations.lift,
            }
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Report':
        scores = data['scores']
        tables = data['tabels']
        visualizations = data['visualizations']

        return Report(
            scores=Report.Scores(**scores),
            tables=Report.Tables(
                confusion_matrix=np.array(tables['confusion_matrix']),
                classification_report=tables['classification_report']
            ),
            visualizations=Report.Visualizations(**visualizations)
        )

    @staticmethod
    def average_reports(reports: List['Report']) -> 'Report':
        if not reports:
            raise ValueError("List of reports is empty")

        avg_scores = Report.Scores(
            accuracy=np.mean([r.scores.accuracy for r in reports]),
            cohen_kappa=np.mean([r.scores.cohen_kappa for r in reports]),
            mcc=np.mean([r.scores.mcc for r in reports]),
            specificity=np.mean([r.scores.specificity for r in reports])
        )

        avg_confusion_matrix = np.mean([r.tables.confusion_matrix for r in reports], axis=0)

        # Average classification report (only numeric values)
        avg_classification_report = {}
        for key in reports[0].tables.classification_report.keys():
            if isinstance(reports[0].tables.classification_report[key], dict):
                avg_classification_report[key] = {
                    k: np.mean([r.tables.classification_report[key][k] for r in reports])
                    for k in reports[0].tables.classification_report[key].keys()
                }
            elif key != 'accuracy':  # Skip 'accuracy' as it's already in scores
                avg_classification_report[key] = np.mean([r.tables.classification_report[key] for r in reports])

        # Average visualizations
        avg_visualizations = Report.Visualizations(
            roc=Report._average_roc_curve([r.visualizations.roc for r in reports]),
            precision_recall=Report._average_precision_recall_curve(
                [r.visualizations.precision_recall for r in reports]),
            cumulative_gain=Report._average_cumulative_gain_curve([r.visualizations.cumulative_gain for r in reports]),
            lift=Report._average_lift_curve([r.visualizations.lift for r in reports])
        )

        return Report(
            scores=avg_scores,
            tables=Report.Tables(
                confusion_matrix=avg_confusion_matrix,
                classification_report=avg_classification_report
            ),
            visualizations=avg_visualizations
        )

    @staticmethod
    def _average_roc_curve(roc_curves: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Interpolate ROC curves to a common set of false positive rates
        common_fpr = np.linspace(0, 1, 100)
        interpolated_tpr = []

        for roc in roc_curves:
            fpr, tpr = roc['fpr'], roc['tpr']
            interpolated_tpr.append(np.interp(common_fpr, fpr, tpr))

        avg_tpr = np.mean(interpolated_tpr, axis=0)
        avg_auc = np.mean([roc['auc'] for roc in roc_curves])

        return {'fpr': common_fpr, 'tpr': avg_tpr, 'auc': np.array([avg_auc])}

    @staticmethod
    def _average_precision_recall_curve(pr_curves: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Interpolate precision-recall curves to a common set of recall values
        common_recall = np.linspace(0, 1, 100)
        interpolated_precision = []

        for pr in pr_curves:
            recall, precision = pr['recall'], pr['precision']
            interpolated_precision.append(np.interp(common_recall, recall[::-1], precision[::-1])[::-1])

        avg_precision = np.mean(interpolated_precision, axis=0)

        return {'precision': avg_precision, 'recall': common_recall}

    @staticmethod
    def _average_cumulative_gain_curve(cg_curves: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Interpolate cumulative gain curves to a common set of percentages
        common_percentages = np.linspace(0, 1, 100)
        interpolated_gains1 = []
        interpolated_gains2 = []

        for cg in cg_curves:
            percentages, gains1, gains2 = cg['percentages'], cg['gains1'], cg['gains2']
            interpolated_gains1.append(np.interp(common_percentages, percentages, gains1))
            interpolated_gains2.append(np.interp(common_percentages, percentages, gains2))

        avg_gains1 = np.mean(interpolated_gains1, axis=0)
        avg_gains2 = np.mean(interpolated_gains2, axis=0)

        return {'gains1': avg_gains1, 'gains2': avg_gains2, 'percentages': common_percentages}

    @staticmethod
    def _average_lift_curve(lift_curves: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Interpolate lift curves to a common set of percentages
        common_percentages = np.linspace(0, 1, 100)
        interpolated_gains1 = []
        interpolated_gains2 = []

        for lift in lift_curves:
            percentages, gains1, gains2 = lift['percentages'], lift['gains1'], lift['gains2']
            interpolated_gains1.append(np.interp(common_percentages, percentages, gains1))
            interpolated_gains2.append(np.interp(common_percentages, percentages, gains2))

        avg_gains1 = np.mean(interpolated_gains1, axis=0)
        avg_gains2 = np.mean(interpolated_gains2, axis=0)

        return {'gains1': avg_gains1, 'gains2': avg_gains2, 'percentages': common_percentages}

    @staticmethod
    def _predict_proba(results: np.ndarray) -> np.ndarray:
        return np.column_stack((1 - results, results))

    @staticmethod
    def _calculate_specificity(matrix: np.ndarray) -> float:
        tn, fp, _, _ = matrix.ravel()
        return tn / (tn + fp)

    @staticmethod
    def _calculate_roc_curve(y: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)
        return {'fpr': fpr, 'tpr': tpr, 'auc': np.array([metrics.auc(fpr, tpr)])}

    @staticmethod
    def _calculate_precision_recall_curve(y: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        precision, recall, _ = metrics.precision_recall_curve(y, y_pred_proba)
        return {'precision': precision, 'recall': recall}

    @staticmethod
    def _calculate_cumulative_gain_curve(y: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0])
        _, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1])
        return {'gains1': gains1, 'gains2': gains2, 'percentages': percentages}

    @staticmethod
    def _calculate_lift_curve(y: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        classes = np.unique(y)
        percentages, gains1 = cumulative_gain_curve(y, y_pred_proba[:, 0], classes[0])
        _, gains2 = cumulative_gain_curve(y, y_pred_proba[:, 1], classes[1])

        percentages = percentages[1:]
        gains1 = gains1[1:] / percentages
        gains2 = gains2[1:] / percentages

        return {'gains1': gains1, 'gains2': gains2, 'percentages': percentages}
