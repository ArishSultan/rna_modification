from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    return list(
        sequence[0: j + 1].count(sequence[j]) / (j + 1)
        for j in range(len(sequence))
    )


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'anf')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
