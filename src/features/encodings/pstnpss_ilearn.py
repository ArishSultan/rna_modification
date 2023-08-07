from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import encode_df
from ...data import Species


def encode(sequence: str, species: Species = Species.human) -> list[float]:
    """
    Calculates the differential codon usage bias in a given sequence, between a set of positive and negative samples for a given species.

    Mathematically, let P_{i,j} represent the count of the j-th codon at position i in the positive samples,
    N_{i,j} represent the count of the j-th codon at position i in the negative samples, S_{p} represent the
    total size of the positive sample set, and S_{n} represent the total size of the negative sample set. Then,
    the differential codon usage at position i for the j-th codon can be calculated as:

    DU_{i,j} = P_{i,j}/S_{p} - N_{i,j}/S_{n}

    :param sequence: Input DNA sequence
    :param species: Name of the species ('hs', 'sc', or 'mm')
    :return: A tuple containing the differential codon usage for each codon in the sequence.
    """
    match species:
        case Species.human:
            neg_mat = HS_NEG
            pos_mat = HS_POS
            pos_size = 4327
            neg_size = 3732
        case Species.yeast:
            neg_mat = SC_NEG
            pos_mat = SC_POS
            pos_size = 1147
            neg_size = 1147
        case Species.mouse:
            neg_mat = MM_NEG
            pos_mat = MM_POS
            pos_size = 3174
            neg_size = 3174
        case _:
            return [.0] * 19

    return list(
        (pos_mat[i][ORDER[sequence[i: i + 3]]] / pos_size - neg_mat[i][ORDER[sequence[i: i + 3]]] / neg_size) for i in
        range(len(sequence) - 2)
    )


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, species: Species):
        self._species = species

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self._species), 'pstnpss')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return self.fit_transform(x)
