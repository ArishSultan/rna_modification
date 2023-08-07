import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.features import encode_df
from ...data.seq_bunch import SeqBunch


def encode(sequence: str, pos_matrix: list, neg_matrix: list) -> list[float]:
    pass


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, bunch: SeqBunch, **kwargs) -> DataFrame:
        print(bunch.samples)    

        pass
        # return encode_df(x, lambda seq: encode(seq, self._species), 'pstnpss')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return self.fit_transform(x)


def CalculateMatrix(data, order):
    matrix = np.zeros((len(data[0]) - 2, 64))
    for i in range(len(data[0]) - 2):
        for j in range(len(data)):
            matrix[i][order[data[j][i:i + 3]]] += 1

    return matrix
