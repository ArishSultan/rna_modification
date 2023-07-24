from sklearn.base import TransformerMixin

from sklearn.model_selection import KFold

from .report import Report
from ..model import ModelFactory
from ..data.seq_bunch import SeqBunch


class Experiment:
    def __init__(self, factory: ModelFactory, test: SeqBunch, train: SeqBunch, encoding: TransformerMixin, k=5):
        self.k = k
        self.test = test
        self.train = train
        self.factory = factory
        self.encoding = encoding

    def run(self):
        k_fold = KFold(n_splits=self.k, random_state=42, shuffle=True)

        y = self.train.targets
        x = self.encoding.fit_transform(self.train.samples)

        k_fold_reports = []
        for train_index, test_index in k_fold.split(x):
            train_x, test_x = x.iloc[train_index], x.iloc[test_index]
            train_y, test_y = y.iloc[train_index], y.iloc[test_index]

            model = self.factory.create_model()
            model.fit(train_x, train_y)

            k_fold_reports.append(Report.create_report(model, (test_x, test_y)))

        test_y = self.test.targets
        test_x = self.encoding.fit_transform(self.test.samples)

        model = self.factory.create_model()
        model.fit(x, y)

        return {
            'train': k_fold_reports,
            'test': Report.create_report(model, (test_x, test_y))
        }
