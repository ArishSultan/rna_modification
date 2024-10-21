from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df

from collections import Counter


def encode(sequence: str, window: int = 5) -> list[float]:
    encoded_seq = []

    nucleotides = 'ACGU'
    for j in range(len(sequence)):
        if j < len(sequence) and j + window <= len(sequence):
            count = Counter(sequence[j:j + window])

            for nucleotide in nucleotides:
                encoded_seq.append(count[nucleotide] / len(sequence[j:j + window]))

    return encoded_seq


class Encoder(BaseEncoder):
    def __init__(self, window: int = 5):
        self.window = window

    def fit(self, x: DataFrame, y: Series):
        print('ENAC encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.window), f'enac_{self.window}')

    def transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return self.fit_transform(x)
