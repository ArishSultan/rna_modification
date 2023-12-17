from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer
from ...utils.features import encode_df


def encode(sequence: str, k: int = 2) -> list[float]:
    return kmer.encode(sequence, k, upto=False, normalize=True)


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 2):
        self.k = k

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.k), f'knc_{self.k}')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
