from pandas import DataFrame, Series

from . import kmer
from ..encoder import BaseEncoder
from ...utils.features import encode_df


def encode(sequence: str, k: int = 2) -> list[float]:
    return kmer.encode(sequence, k, upto=False, normalize=True)


class Encoder(BaseEncoder):
    def __init__(self, k: int = 2):
        self.k = k

    def fit(self, x: DataFrame, y: Series):
        print('KNC encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.k), f'knc_{self.k}')

    def transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return self.fit_transform(x)
