from sklearn.model_selection import KFold

from .report import Report
from ..model import ModelFactory
from ..data.seq_bunch import SeqBunch


class Experiment:
    def __init__(self, factory: ModelFactory, test: SeqBunch | None, train: SeqBunch, encoding, k=5):
        self.k = k
        self.test = test
        self.train = train
        self.factory = factory
        self.encoding = encoding

    def run(self):
        k_fold = KFold(n_splits=self.k, random_state=42, shuffle=True)

        x = self.encoding.fit_transform(self.train.samples, y=self.train.targets)
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

        test_x = self.encoding.transform(self.test.samples)
        test_y = self.test.targets

        model = self.factory.create_model()
        model.fit(x, y)

        return {
            'train': k_fold_reports,
            'test': Report.create_report(model, (test_x, test_y))
        }


class ExperimentNew:
    def __init__(self, factory: ModelFactory, test: SeqBunch | None, train: SeqBunch, encoding, k=5):
        self.k = k
        self.test = test
        self.train = train
        self.factory = factory
        self.encoding = encoding

    def run(self):
        k_fold = KFold(n_splits=self.k, random_state=42, shuffle=True)

        encoded_train = self.encoding.fit_transform(self.train)
        y = encoded_train.targets
        x = encoded_train.samples

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

        encoded_test = self.encoding.transform(self.test)
        test_y = encoded_test.targets
        test_x = encoded_test.samples

        model = self.factory.create_model()
        model.fit(x, y)

        return {
            'train': k_fold_reports,
            'test': Report.create_report(model, (test_x, test_y))
        }
