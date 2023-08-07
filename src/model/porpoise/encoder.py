from pandas import DataFrame
from operator import itemgetter
from sklearn.base import BaseEstimator, TransformerMixin

from . import pstnpss
from ...data import Species
from ...utils.features import encode_df
from ...features.encodings import pse_knc, binary


def encode(sequence: str, species: Species, pse_knc_info=None):
    if pse_knc_info is None:
        pse_knc_info = pse_knc.get_info()

    x0 = list(itemgetter(
        10, 12, 8, 9, 17, 7, 6, 15, 14, 11, 2, 13, 5, 16, 18, 4, 1, 3, 0
    )(pstnpss.encode(sequence, species)))
    x1 = list(itemgetter(
        4, 21, 20, 00, 5, 65, 64, 1, 17, 16, 60, 15, 25, 54, 3, 11, 46, 41, 40, 39, 7,
        2, 42, 36, 31, 12, 24, 35, 18, 62, 61, 53, 8, 50, 56, 43, 19, 28, 9, 59, 37, 38,
        33, 48, 44, 45, 63, 10, 22, 27, 49, 57, 55, 14, 51, 47, 52, 26, 23, 29, 30, 13, 6, 58
    )(pse_knc.encode(sequence, pse_knc_info, 3, 2, 0.1)))
    x2 = list(itemgetter(46, 36, 44, 77, 45, 49, 3)(binary.encode(sequence)))

    return x0 + x0 + x1 + x2


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, species: Species, pse_knc_info=None):
        self.species = species

        if pse_knc_info is None:
            self.pse_knc_info = pse_knc.get_info()

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.species, self.pse_knc_info), 'porpoise')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
