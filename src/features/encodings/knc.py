from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer
from ...data.seq_bunch import SeqBunch
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
    def __init__(self, k: int = 2):
        self.k = k

    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:
        return SeqBunch(
            targets=bunch.targets,
            description=bunch.description,
            samples=encode_df(bunch.samples, lambda seq: encode(seq, self.k), f'knc_{self.k}')
        )

    def transform(self, x: SeqBunch) -> SeqBunch:
        return self.fit_transform(x)
