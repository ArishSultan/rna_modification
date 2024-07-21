from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    return [encode_nucleotide(sequence, i) for i in range(len(sequence))]


def encode_nucleotide(sequence, index) -> float:
    return sequence[:index].count(sequence[index]) / (index + 1)


class Encoder(BaseEncoder):
    def fit(self, x: DataFrame, y: Series):
        print('ND encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'nd')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
