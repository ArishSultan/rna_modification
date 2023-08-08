from itertools import product
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

from ...data.seq_bunch import SeqBunch
from ...utils.features import encode_df


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
    all_mers = generate_all_kmers(k, upto)

    counter = Counter()
    counter.update(seq_mers)

    result = dict()
    for mer in all_mers:
        if mer in counter:
            if normalize:
                result[mer] = float(counter[mer] / len(seq_mers))
            else:
                result[mer] = float(counter[mer])
        else:
            result[mer] = 0.0

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

    if upto:
        repeats = list(range(1, k + 1))
    else:
        repeats = [k]

    results = list[str]()
    for i in repeats:
        results += [
            sequence[j: j + i]
            for j in range(len(sequence) - i + 1)
        ]

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

    if upto:
        repeats = list(range(1, k + 1))
    else:
        repeats = [k]

    results = list[str]()
    for i in repeats:
        results += [''.join(x) for x in product(nucleotides, repeat=i)]

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

    seq_mers = seq_to_kmer(sequence, k, upto)
    all_mers = generate_all_kmers(k, upto)

    counter = Counter()
    counter.update(seq_mers)

    result = list[float]()
    for mer in all_mers:
        if mer in counter:
            if normalize:
                result.append(counter[mer] / len(seq_mers))
            else:
                result.append(counter[mer])
        else:
            result.append(0)

    return result


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 2, upto: bool = False, normalize: bool = False):
        self.k = k
        self.upto = upto
        self.normalize = normalize

    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:
        return SeqBunch(
            targets=bunch.targets,
            description=bunch.description,
            samples=encode_df(bunch.samples, lambda seq: encode(seq, self.k, self.upto, self.normalize), 'kmer')
        )

    def transform(self, x: SeqBunch) -> SeqBunch:
        return self.fit_transform(x)
