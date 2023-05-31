from pandas import DataFrame
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import encode_df


def create_ps_dict(k: int) -> dict:
    """
    Create a dictionary for k-mers and their position specific binary representation.

    :param k: Length of the k-mers.
    :return: A dictionary with k-mers as keys and binary representation as values.
    """
    kmers = [''.join(x) for x in product('ACGU', repeat=k)]
    return {kmer: [int(i == index) for i in range(len(kmers))] for index, kmer in enumerate(kmers)}


def encode(sequence: str, k: int, ps_dict: dict | None = None) -> list[float]:
    """
    Position Specific (PS) encoding of a sequence.

    Given a sequence s = s_1s_2...s_n, the PS encoding of s is the concatenation of v_{s_i...s_{i+k-1}}
    for i = 1 to n-k+1, where v_k is a binary vector representing k-mer k.

    :param sequence: The sequence to be encoded.
    :param k: The length of k-mers.
    :param ps_dict: Optional precomputed dictionary for position specific encoding.
    :return: A tuple representing the PS encoding of the sequence.
    """
    if ps_dict is None:
        ps_dict = create_ps_dict(k)
    return [val for subseq in (ps_dict.get(sequence[i:i + k], [0] * 4 ** k) for i in range(len(sequence) - k + 1)) for
            val
            in subseq]


class Encoder(BaseEstimator, TransformerMixin):
    """
    A transformer that applies the ANF encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the ANF
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the ANF of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import anf
    >>> encoder = anf.Encoder()
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def __init__(self, k=2):
        self.k = k

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        ps_dict = create_ps_dict(self.k)
        return encode_df(x, lambda seq: encode(seq, self.k, ps_dict), f'ps_{self.k}')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return self.fit_transform(x)
