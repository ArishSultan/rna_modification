from itertools import product
from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def create_ps_dict(k: int) -> dict:
    kmers = [''.join(x) for x in product('ACGU', repeat=k)]
    return {kmer: [int(i == index) for i in range(len(kmers))] for index, kmer in enumerate(kmers)}


def encode(sequence: str, k: int, ps_dict: dict | None = None) -> list[float]:
    if ps_dict is None:
        ps_dict = create_ps_dict(k)
    return [val for subseq in (ps_dict.get(sequence[i:i + k], [0] * 4 ** k) for i in range(len(sequence) - k + 1)) for
            val
            in subseq]


class Encoder(BaseEncoder):
    def __init__(self, k=2):
        self.k = k

    def fit(self, x: DataFrame, y: Series):
        print('PS encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        ps_dict = create_ps_dict(self.k)
        return encode_df(x, lambda seq: encode(seq, self.k, ps_dict), f'ps_{self.k}')

    def transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return self.fit_transform(x)
