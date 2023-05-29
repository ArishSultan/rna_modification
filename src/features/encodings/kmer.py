from pandas import DataFrame
from itertools import product
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


def kmer_count(sequence: str, k: int = 2, upto: bool = False, normalize: bool = False) -> dict:
    """
    Given a sequence, this function returns a dictionary of k-mer counts.

    Args:
    sequence (str): Input DNA/RNA sequence
    k (int): Length of k-mer
    upto (bool): If True, counts all k-mers of lengths upto k
    normalize (bool): If True, returns frequency instead of count

    Returns:
    dict: A dictionary of k-mer and its count/frequency
    """

    seq_mers = seq_to_kmer(sequence, k, upto)
    counter = Counter(seq_mers)

    all_mers = generate_all_kmers(k, upto)
    result = {mer: (counter.get(mer, 0.0) / len(seq_mers) if normalize else counter.get(mer, 0.0)) for mer in all_mers}

    return result


def seq_to_kmer(sequence: str, k: int = 2, upto: bool = False) -> tuple[str]:
    """
    Given a sequence and k, this function returns all k-mers in the sequence.

    Args:
    sequence (str): Input DNA/RNA sequence
    k (int): Length of k-mer
    upto (bool): If True, returns all k-mers of lengths upto k

    Returns:
    tuple[str]: Tuple of all k-mers

    Raises:
    Exception: If k is less than 1
    """

    if k < 1:
        raise Exception("[k] must be greater then 0")

    repeats = range(1, k + 1) if upto else [k]

    results = [sequence[j: j + i] for i in repeats for j in range(len(sequence) - i + 1)]

    return tuple(results)


def generate_all_kmers(k: int = 2, upto: bool = False) -> tuple[str]:
    """
    Generate all possible k-mers of either DNA or RNA.

    Args:
    k (int): Length of k-mer
    upto (bool): If True, generates all k-mers of lengths upto k

    Returns:
    tuple[str]: Tuple of all possible k-mers

    Raises:
    AssertionError: If k is less than 1
    """
    assert k > 0
    nucleotides = 'ACGU'

    repeats = range(1, k + 1) if upto else [k]

    results = [''.join(x) for i in repeats for x in product(nucleotides, repeat=i)]

    return tuple(results)


def encode(sequence: str, k: int = 2, upto: bool = False, normalize: bool = False) -> list[float]:
    """
    Encodes DNA / RNA sequence to k-mer count/frequency format.

    The k-mer encoding is mathematically defined as follows: $$\Phi_K(S) = (s_1s_2...s_K, s_2s_3...s_{K+1}, ...,
    s_{N-K+1}...s_{N-1}s_N)$$

    If we consider K-mer encoding as counting the number of each specific K-mer in the sequence, the mathematical
    expression could be something like:

    $$\text{For each possible K-mer } k, C_k = \sum_{i=1}^{N-K+1} I(k_i = k)$$where $I$ is the indicator function that
    is $1$ if $k_i = k$ and $0$ otherwise, and $C_k$ is the count of k-mer $k$ in the sequence. If normalize is True,
    the function returns frequencies instead of counts: For each possible k-mer $k$, $F_k = \frac{C_k}{\sum_{k'} C_{k'}
    }$, where $F_k$ is the frequency of k-mer $k$, $C_k$ is the count of k-mer $k$, and the denominator $\sum_{k'} C_{k'
    }$ is the total count of all k-mers in the sequence.

    Args:
    sequence (str): The DNA/RNA sequence S = {s1, s2, ..., sN} where si is the nucleotide at the i-th position and N is
                    the length of the sequence.

    k (int): Length of k-mer
    upto (bool): If True, generates all k-mers of lengths upto k
    normalize (bool): If True, returns frequency instead of count

    Returns:
    tuple[float]: A sequence K = {k1, k2, ..., kN-k+1} of k-mer counts or frequencies where each ki is a count or
                  frequency of the i-th k-mer in the original sequence.
    """

    return list(kmer_count(sequence, k, upto, normalize).values())


class Encoder(BaseEstimator, TransformerMixin):
    """
    A transformer that applies the Kmer encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the Kmer
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the Kmer of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import kmer
    >>> encoder = kmer.Encoder(k=3)
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def __init__(self, k: int = 2, upto: bool = False, normalize: bool = False):
        """
        Creates an instance of Kmer Encoder with default parameters

        Args:
        k (int): Length of k-mer
        upto (bool): If True, generates all k-mers of lengths upto k
        normalize (bool): If True, returns frequency instead of count
        """
        self.k = k
        self.upto = upto
        self.normalize = normalize

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Implementation of base fit_transform.

        Since, there is nothing in `kmer` encoding that needs fitting so, it just
        transforms all the sequences to their `kmer` encoding.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Kmer-encoded sequences.
        """
        return x.applymap(lambda seq: encode(seq, self.k, self.upto, self.normalize))

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Kmer-encoded sequences.
        """
        return self.fit_transform(x)
