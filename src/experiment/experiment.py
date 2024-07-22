from typing import Callable, Any
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .reports import Report
from ..model import ModelFactory
from ..dataset.seq_bunch import SeqBunch
from ..features.encoder import BaseEncoder


class Experiment:
    def __init__(self, factory: ModelFactory, test: SeqBunch | None, train: SeqBunch, encoding, k=5,
                 should_fit_encoder=True):
        self.k = k
        self.test = test
        self.train = train
        self.factory = factory
        self.encoding = encoding
        self.should_fit_encoder = should_fit_encoder

    def run(self):
        k_fold = KFold(n_splits=self.k, random_state=42, shuffle=True)

        if self.encoding is not None:
            if self.should_fit_encoder:
                x = self.encoding.fit_transform(self.train.samples, y=self.train.targets)
            else:
                x = self.encoding.transform(self.train.samples)
        else:
            x = self.train.samples
        y = self.train.targets

        k_fold_reports = []
        for train_index, test_index in k_fold.split(x):
            train_x, test_x = x.iloc[train_index], x.iloc[test_index]
            train_y, test_y = y.iloc[train_index], y.iloc[test_index]

            model = self.factory.create_model()
            model.fit(train_x, train_y)

            k_fold_reports.append(Report.create_report(model, (test_x, test_y)))

        if self.test is None:
            return {
                'train': k_fold_reports
            }

        if self.encoding is not None:
            test_x = self.encoding.transform(self.test.samples)
        else:
            test_x = self.test.samples
        test_y = self.test.targets

        model = self.factory.create_model()
        model.fit(x, y)

        return {
            'train': k_fold_reports,
            'test': Report.create_report(model, (test_x, test_y))
        }


class ExperimentNew:
    def __init__(self, model_factory: Callable[[], Any], encoder: BaseEncoder, train_encoder: bool = True):
        self._model = None
        self._encoder = encoder
        self._model_factory = model_factory
        self._train_encoder = train_encoder

    def _fit(self, x: DataFrame, y: Series):
        model = self._model_factory()

        if self._train_encoder:
            self._encoder.fit(x, y)

        return model.fit(self._encoder.transform(x), y)

    def _evaluate(self, model, x: DataFrame, y: Series):
        return Report.create_report(model, (self._encoder.transform(x), y), False)

    def fit(self, x: DataFrame, y: Series):
        self._model = self._fit(x, y)

    def evaluate(self, x: DataFrame, y: Series):
        return self._evaluate(self._model, x, y)

    def cross_validate(self, x: DataFrame, y: Series, k: int = 5) -> tuple[list[Report], Report]:
        skf = StratifiedKFold(n_splits=k)
        reports = []

        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            reports.append(self._evaluate(self._fit(x_train, y_train), x_val, y_val))

        return reports, Report.average_reports(reports)
