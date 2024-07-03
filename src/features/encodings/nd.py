from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    return [encode_nucleotide(sequence, i) for i in range(len(sequence))]


def encode_nucleotide(sequence, index) -> float:
    return sequence[:index].count(sequence[index]) / (index + 1)


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'nd')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
