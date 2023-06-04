from pandas import DataFrame, Series
from sklearn.base import TransformerMixin

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

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
        k_fold = KFold(n_splits=self.k, random_state=42)

        y = self.train.targets
        x = self.encoding.fit_transform(self.train.samples)

        for train_index, test_index in k_fold.split(x):
            train_x, test_x = x[train_index], x[test_index]
            train_y, test_y = y[train_index], y[test_index]

            model = self.factory.create_model()
            model.fit(train_x, train_y)

            # Generate model reports for each k-fold.

        test_y = self.train.targets
        test_x = self.encoding.fit_transform(self.test.samples)

        model = self.factory.create_model()
        model.fit(x, y)

        # Generate a test report
