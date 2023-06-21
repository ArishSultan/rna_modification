import pickle
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer
from ...utils import get_path, encode_df

_K_2_INDEX = {'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
              'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
              'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
              'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15}

_METHODS_DICT = {'DAC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'DCC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'DACC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TAC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TCC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'TACC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PseKNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PCPseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'PCPseTNC': [],
                 'SCPseDNC': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)'],
                 'SCPseTNC': []}

_DATA_FILE_DICT = {'DAC': 'dirnaPhyche.data',
                   'DCC': 'dirnaPhyche.data',
                   'DACC': 'dirnaPhyche.data',
                   'TAC': 'dirnaPhyche.data',
                   'TCC': 'dirnaPhyche.data',
                   'TACC': 'dirnaPhyche.data',
                   'PseDNC': 'dirnaPhyche.data',
                   'PseKNC': 'dirnaPhyche.data',
                   'PCPseDNC': 'dirnaPhyche.data',
                   'PCPseTNC': '',
                   'SCPseDNC': 'dirnaPhyche.data',
                   'SCPseTNC': ''}


def calculate_correlation(mer1, mer2, indexes, values):
    return np.mean([(float(value[indexes[mer1]]) - float(value[indexes[mer2]])) ** 2 for value in values])


def calculate_theta_list(info: tuple, indexes: dict, lambda0: float, sequence: str, k=2):
    keys, values = info
    return [np.mean([calculate_correlation(
        sequence[i:i + k], sequence[i + temp_lambda + 1: i + temp_lambda + 1 + k],
        indexes, values) for i in range(len(sequence) - temp_lambda - k)]
    ) for temp_lambda in range(int(lambda0))]


def get_info(method: str):
    filename = _DATA_FILE_DICT[method]
    with open(get_path(f'data/raw/features/{filename}'), 'rb') as file:
        data = pickle.load(file)
    return _METHODS_DICT[method], data


def encode(sequence: str, info: tuple, k=2, lambda0: float = 1, weight: float = 0) -> list[float]:
    """
    The Pseudo K-tuple Nucleotide Composition (PseKNC) method is a widely used sequence-based feature extraction method. It generates a fixed length feature vector regardless of the sequence length by considering both the occurrence frequency of k-tuples (nucleotide sub-sequences of length k) and the correlation between non-local k-tuples.

    Mathematical Representation:

    For a given sequence S and specific `k` and `lambda`, the PseKNC encoding function Φ returns a sequence of k-mer frequencies and correlations:

    Φ_{PseKNC}(S) = [f_{k_1}, f_{k_2}, ..., f_{k_{4^k}}, θ_1, θ_2, ..., θ_{lambda}]

    where `f_{k_i}` represents the normalized frequency of the i-th k-mer in the sequence S, calculated as the total count of the i-th k-mer in S divided by the total number of k-mers; and `θ_i` represents the correlation between non-local k-mers in S, calculated as:

    θ_i = 1 / (N - i - k + 1) * Σ_{j=1}^{N-i-k+1} (d_{s_js_{j+i}}^2)

    `d_{s_js_{j+i}}` is the Euclidean distance between `s_j` and `s_{j+i}`.

    Parameters:
    sequence (str): The input DNA/RNA sequence (S).
    info (tuple): Additional information for the encoding process.
    k (int): The length of the k-tuple (must be > 0).
    lambda0 (float): The maximum gap between two k-mers (must be >= 0).
    weight (float): The weight factor for the correlations.

    Returns:
    list[float]: The PseKNC encoding of the sequence.
    """
    kmer_count = kmer.kmer_count(sequence, k, normalize=True)
    theta_list = calculate_theta_list(info, _K_2_INDEX, lambda0, sequence, k)
    sum_theta_list = sum(theta_list)

    encoded_seq = [kmer_count[mer] / (1 + weight * sum_theta_list) for mer in kmer.generate_all_kmers(k)]

    base = 4 ** k + 1
    encoded_seq.extend(
        [(weight * theta_list[k - base]) / (1 + weight * sum_theta_list) for k in range(base, base + lambda0)]
    )

    return encoded_seq


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

    def __init__(self, k=2, lambda0: float = 1, weight: float = 0):
        self.k = k
        self.weight = weight
        self.lambda0 = lambda0

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Implementation of base fit_transform.

        Since, there is nothing in `p` encoding that needs fitting so, it just
        transforms all the sequences to their `anf` encoding.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        info = get_info('PseKNC')

        return encode_df(x, lambda seq: encode(seq, info, self.k, self.lambda0, self.weight), 'pse_knc')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return self.fit_transform(x)
