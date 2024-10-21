from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    return list(
        sequence[0: j + 1].count(sequence[j]) / (j + 1)
        for j in range(len(sequence))
    )


class Encoder(BaseEncoder):
    def fit(self, x: DataFrame, y: Series):
        print('ANF encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'anf')

    def transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return self.fit_transform(x)
