import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

TRI_NUCLEOTIDES_DICT = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAU': 3, 'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACU': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGU': 11, 'AUA': 12, 'AUC': 13, 'AUG': 14, 'AUU': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAU': 19, 'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCU': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGU': 27, 'CUA': 28, 'CUC': 29, 'CUG': 30, 'CUU': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAU': 35, 'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCU': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGU': 43, 'GUA': 44, 'GUC': 45, 'GUG': 46, 'GUU': 47,
    'UAA': 48, 'UAC': 49, 'UAG': 50, 'UAU': 51, 'UCA': 52, 'UCC': 53, 'UCG': 54, 'UCU': 55,
    'UGA': 56, 'UGC': 57, 'UGG': 58, 'UGU': 59, 'UUA': 60, 'UUC': 61, 'UUG': 62, 'UUU': 63,
}


def encode(sequence: str, sizes: tuple[int, int], matrices: tuple[list, list]):
    positives_matrix_forward, negatives_matrix_forward, positives_matrix_backward, negatives_matrix_backward = matrices

    new_sample = []
    seq_len = len(sequence)

    for j in range(len(sequence) - 2):
        kmer = sequence[j: j + 3]
        p_size, n_size = sizes

        # Forward direction
        p_number_forward = positives_matrix_forward[j][TRI_NUCLEOTIDES_DICT[kmer]]
        n_number_forward = negatives_matrix_forward[j][TRI_NUCLEOTIDES_DICT[kmer]]

        # Backward direction
        if j + 2 < seq_len:
            kmer_backward = sequence[seq_len - j - 3: seq_len - j]
        else:
            kmer_backward = sequence[seq_len - 3: seq_len]

        p_number_backward = positives_matrix_backward[j][TRI_NUCLEOTIDES_DICT[kmer_backward]]
        n_number_backward = negatives_matrix_backward[j][TRI_NUCLEOTIDES_DICT[kmer_backward]]

        p_number = (p_number_forward + p_number_backward) / 2
        n_number = (n_number_forward + n_number_backward) / 2

        new_sample.append(p_number / p_size - n_number / n_size)
    return new_sample


def _calculate_matrix(data, order, direction='forward'):
    size = len(data[0])
    matrix = np.zeros((size - 2, 64))

    if direction == 'forward':
        for i in range(size - 2):
            for j in range(len(data)):
                matrix[i][order[data[j][i:i + 3]]] += 1
    else:
        for i in range(size - 2):
            for j in range(len(data)):
                kmer = data[j][size - i - 3: size - i]
                matrix[i][order[kmer]] += 1

    return matrix


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, consider_train_target=False, consider_test_target=False):
        self._positive_size = None
        self._negative_size = None
        self._positives_matrix_forward = None
        self._negatives_matrix_forward = None
        self._positives_matrix_backward = None
        self._negatives_matrix_backward = None
        self.consider_test_target = consider_test_target
        self.consider_train_target = consider_train_target

    @property
    def sizes(self) -> tuple[int, int]:
        return self._positive_size, self._negative_size

    @property
    def matrices(self) -> tuple[list, list]:
        return self._positives_matrix_forward, self._negatives_matrix_forward, self._positives_matrix_backward, self._negatives_matrix_backward

    def fit(self, x: DataFrame, y: Series):
        positives = x[y == 1]
        negatives = x[y == 0]

        self._positives_matrix_forward = _calculate_matrix(positives['sequence'].values, TRI_NUCLEOTIDES_DICT,
                                                           direction='forward')
        self._negatives_matrix_forward = _calculate_matrix(negatives['sequence'].values, TRI_NUCLEOTIDES_DICT,
                                                           direction='forward')

        self._positives_matrix_backward = _calculate_matrix(positives['sequence'].values, TRI_NUCLEOTIDES_DICT,
                                                            direction='backward')
        self._negatives_matrix_backward = _calculate_matrix(negatives['sequence'].values, TRI_NUCLEOTIDES_DICT,
                                                            direction='backward')

        self._positive_size = len(positives)
        self._negative_size = len(negatives)

    def fit_transform(self, x: DataFrame, y: Series, **kwargs) -> DataFrame:
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x: DataFrame) -> DataFrame:
        if self._positive_size is None:
            raise Exception(
                'Call `fit` before calling `transform` because this encoding needs collective sequence information')

        sequences = (x['sequence'] if 'sequence' in x else x[0]).values

        new_samples = []
        for i in range(len(sequences)):
            new_samples.append(encode(sequences[i], self.sizes, self.matrices))

        return DataFrame(
            data=new_samples,
            columns=[f'bipstp_{i}' for i in range(len(self._positives_matrix_forward))]
        )
