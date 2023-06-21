from abc import ABC, abstractmethod

from .report import Report


class KFoldReport(ABC):
    def __init__(self, reports: list[Report]):
        self._reports = reports

    @abstractmethod
    def plot_roc(self):
        pass

    @abstractmethod
    def plot_lift(self):
        pass

    @abstractmethod
    def plot_cumulative_gain(self):
        pass

    @abstractmethod
    def plot_precision_recall(self):
        pass

    @abstractmethod
    def display_confusion_matrix(self):
        pass

    @abstractmethod
    def display_classification_report(self):
        pass

    @abstractmethod
    def display_scores(self):
        pass
