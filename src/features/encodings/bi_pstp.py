import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

from .kmer import generate_all_kmers
from ...utils.features import encode_df


def _valid_pairs(sequence: str, k: int, gap: int):
    size = len(sequence)
    reversed_sequence = sequence[::-1]
    assert k > 0 and 0 <= gap <= (size - 5) / 2

    forward_pairs = list()
    backward_pairs = list()
    for i in range(gap + k - 1, size - gap - k + 1):
        forward_pairs.append(sequence[i] + sequence[i + gap + 1: i + gap + k])
        backward_pairs.append(reversed_sequence[i] + reversed_sequence[i + gap + 1: i + gap + k])

    return forward_pairs, backward_pairs[::-1]


def _pair_nucleotides(sequence: str, k: int, gap: int):
    size = len(sequence)
    reversed_sequence = sequence[::-1]

    assert k > 0 and 0 <= gap <= (size - 5) / 2

    forward_pairs = list()
    backward_pairs = list()
    for i in range(size - k - gap + 1):
        forward_pairs.append(sequence[i] + sequence[i + gap + 1: i + gap + k])
        backward_pairs.append(reversed_sequence[i] + reversed_sequence[i + gap + 1: i + gap + k])

    return forward_pairs, backward_pairs[::-1]


def _init_empty_matrix(size: int, k: int):
    return np.zeros((size, 4 ** k))


def _init_kmers_index_dict(k):
    return {kmer: idx for idx, kmer in enumerate(generate_all_kmers(k))}


def _calculate_bi_propensity_matrices(sequences: list[str], k: int, gap: int, kmers_dict: dict):
    seq_size = len(sequences[0])
    matrix_forwards = _init_empty_matrix(seq_size, k)
    matrix_backwards = _init_empty_matrix(seq_size, k)

    for sequence in sequences:
        forward_pairs, backward_pairs = _pair_nucleotides(sequence, k, gap)

        for i in range(seq_size - gap - k + 1):
            matrix_forwards[i][kmers_dict[forward_pairs[i]]] += 1
            matrix_backwards[i + k - 1][kmers_dict[forward_pairs[i]]] += 1

    matrix_forwards = matrix_forwards.T
    matrix_backwards = matrix_backwards.T

    # TODO: Consider this probability calculation function
    # for i in range(len(matrix_forwards)):
    #     matrix_forwards[i] /= np.sum(matrix_forwards[i])
    #     matrix_backwards[i] /= np.sum(matrix_backwards[i])

    for i in range(len(matrix_forwards)):
        matrix_forwards[i] /= len(sequences)
        matrix_backwards[i] /= len(sequences)

    return matrix_forwards, matrix_backwards


def _bi_encode_kmers(f_kmer, b_kmer, size, k, gap, matrices, kmers_dict) -> float:
    forward_matrix, backward_matrix = matrices

    return (forward_matrix[kmers_dict[f_kmer]][gap + k - 1] +
            backward_matrix[kmers_dict[b_kmer]][size - gap - k + 1]) / 2


def _encode_internal(sequence, k: int, gap: int, matrices, kmers_dict):
    size = len(sequence)
    forward_pairs, backward_pairs = _valid_pairs(sequence, k, gap)

    pos_result = np.zeros(len(forward_pairs))
    neg_result = np.zeros(len(forward_pairs))
    for i in range(len(forward_pairs)):
        f_kmer = forward_pairs[i]
        b_kmer = backward_pairs[i]

        pos_result[i] = _bi_encode_kmers(f_kmer, b_kmer, size, k, gap, matrices[0], kmers_dict)
        neg_result[i] = _bi_encode_kmers(f_kmer, b_kmer, size, k, gap, matrices[1], kmers_dict)

    return pos_result - neg_result


def _encode(sequence, k: int, min_gap: int, max_gap: int, gap_matrices: dict, kmers_dict: dict):
    return np.concatenate([
        _encode_internal(sequence, k, gap, gap_matrices[gap], kmers_dict) for gap in range(min_gap, max_gap)
    ])


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, k: int, gap: int, min_gap: int = 0):
        assert 0 <= min_gap <= gap and 0 <= gap

        self._k = k
        self._gap = gap
        self._min_gap = min_gap
        self._kmers_dict = dict()
        self._gap_matrices = dict()

    def fit(self, x: DataFrame, y: Series):
        self._gap_matrices.clear()

        pos_samples = x[y == 1]
        neg_samples = x[y == 0]
        self._kmers_dict = _init_kmers_index_dict(self._k)

        for gap in range(self._min_gap, self._gap + 1):
            self._gap_matrices[gap] = (
                _calculate_bi_propensity_matrices(pos_samples['sequence'].values, self._k, gap, self._kmers_dict),
                _calculate_bi_propensity_matrices(neg_samples['sequence'].values, self._k, gap, self._kmers_dict)
            )

    def transform(self, x: DataFrame):
        if len(self._gap_matrices) == 0:
            raise 'Call `fit` before calling `transform` because this encoding needs collective sequence information'

        return encode_df(x, lambda seq: _encode(seq, self._k, self._min_gap, self._gap + 1, self._gap_matrices,
                                                self._kmers_dict), 'bipstp')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        self.fit(x, kwargs['y'])
        return self.transform(x)
