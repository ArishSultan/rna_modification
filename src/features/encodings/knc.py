from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer
from ...utils.features import encode_df


def encode(sequence: str, k: int = 2) -> list[float]:
    """
    The K-tuple Nucleotide Composition (KNC) encoding method represents DNA or RNA sequences by
    counting the normalized frequency of each possible k-tuple (subsequence of k nucleotides) within the sequence.

    This function applies the KNC encoding to a given sequence. It uses a sliding window of size `k`
    that moves along the sequence, one nucleotide at a time. At each position, it extracts the k-tuple and increments
    its count in a count vector. The resulting count vector represents the normalized frequency of occurrence of
    each possible k-tuple in the sequence.

    Mathematical Representation:

    For a given sequence $S = s_1, s_2, ..., s_N$ and a specific `k`, the KNC encoding function Φ returns a sequence of
    k-mer frequencies:

    Φ_{KNC}(S) = \frac{1}{N-K+1} (f_{k_1}, f_{k_2}, ..., f_{k_{N-K+1}})

    where $f_{k_i}$ represents the frequency of the i-th k-mer in the sequence S, calculated as the total count of the
    i-th k-mer in S divided by the total number of k-mers (N-K+1).

    Parameters:
    sequence (str): The input DNA/RNA sequence (S).
    k (int): The length of the k-tuple (must be > 0).

    Returns:
    tuple[float]: A tuple representing the normalized count (frequency) of each k-tuple in the sequence.
    """
    return kmer.encode(sequence, k, upto=False, normalize=True)


class Encoder(BaseEstimator, TransformerMixin):
    """
    A transformer that applies the Knc encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the KNC
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the KNC of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import knc
    >>> encoder = knc.Encoder(k=3)
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def __init__(self, k: int = 2):
        """
        Creates an instance of KNC Encoder with default parameters

        Args:
        k (int): The length of the k-tuple (must be > 0).
        """
        self.k = k

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Implementation of base fit_transform.

        Since, there is nothing in `knc` encoding that needs fitting so, it just
        transforms all the sequences to their `knc` encoding.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of KNC-encoded sequences.
        """
        return encode_df(x, lambda seq: encode(seq, self.k), f'knc_{self.k}')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of KNC-encoded sequences.
        """
        return self.fit_transform(x)
