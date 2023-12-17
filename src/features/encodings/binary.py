from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    nucleotide_codes = {'A': [1.0, 0.0, 0.0, 0.0],
                        'C': [0.0, 1.0, 0.0, 0.0],
                        'G': [0.0, 0.0, 1.0, 0.0],
                        'T': [0.0, 0.0, 0.0, 1.0],
                        'U': [0.0, 0.0, 0.0, 1.0]}

    encoded_seq = []

    for nucleotide in sequence:
        encoded_seq += nucleotide_codes.get(nucleotide, [0.0, 0.0, 0.0, 0.0])

    return encoded_seq


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'binary')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
