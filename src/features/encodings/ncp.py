from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    encoded_seq = list[float]()

    for nucleotide in sequence:
        encoded_seq += encode_nucleotide(nucleotide)

    return encoded_seq


def encode_nucleotide(nucleotide: str) -> list[float]:
    match nucleotide:
        case 'A':
            return [1.0, 1.0, 1.0]
        case 'T':
            return [0.0, 0.0, 1.0]
        case 'U':
            return [0.0, 0.0, 1.0]
        case 'G':
            return [1.0, 0.0, 0.0]
        case 'C':
            return [0.0, 1.0, 0.0]


class Encoder(BaseEncoder):
    def fit(self, x: DataFrame, y: Series):
        print('NCP encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'ncp')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
