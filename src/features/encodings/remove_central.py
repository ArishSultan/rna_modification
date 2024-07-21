from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def encode(sequence: str) -> str:
    mid = len(sequence) // 2
    return sequence[:mid] + sequence[mid + 1:]


class Encoder(BaseEncoder):
    def fit(self, x: DataFrame, y: Series):
        print('Remove Central encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'sequence')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
