import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...data.seq_bunch import SeqBunch

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


def encode(sequence: str, sizes: tuple[int, int], matrices: tuple[list, list], target: int | None):
    positives_matrix, negatives_matrix = matrices

    new_sample = []
    for j in range(len(sequence) - 2):
        kmer = sequence[j: j + 3]
        p_size, n_size = sizes

        p_number = positives_matrix[j][TRI_NUCLEOTIDES_DICT[kmer]]
        if target is not None:
            if target == 1 and p_number > 0:
                p_size -= 1
                p_number -= 1

        n_number = negatives_matrix[j][TRI_NUCLEOTIDES_DICT[kmer]]
        if target is not None:
            if target == 0 and n_number > 0:
                n_size -= 1
                n_number -= 1

        new_sample.append(p_number / p_size - n_number / n_size)
    return new_sample


def _calculate_matrix(data, order):
    size = len(data[0])

    matrix = np.zeros((size - 2, 64))
    for i in range(size - 2):
        for j in range(len(data)):
            matrix[i][order[data[j][i:i + 3]]] += 1

    return matrix


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._positive_size = None
        self._negative_size = None
        self._positives_matrix = None
        self._negatives_matrix = None

    @property
    def sizes(self) -> tuple[int, int]:
        return self._positive_size, self._negative_size

    @property
    def matrices(self) -> tuple[list, list]:
        return self._positives_matrix, self._negatives_matrix

    def fit(self, bunch: SeqBunch):
        positives = bunch.samples[bunch.targets == 1]
        negatives = bunch.samples[bunch.targets == 0]

        self._positives_matrix = _calculate_matrix(positives['sequence'].values, TRI_NUCLEOTIDES_DICT)
        self._negatives_matrix = _calculate_matrix(negatives['sequence'].values, TRI_NUCLEOTIDES_DICT)

        self._positive_size = len(positives)
        self._negative_size = len(negatives)

    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:
        self.fit(bunch)
        return self.transform(bunch, consider_target=False)

    def transform(self, bunch: SeqBunch, consider_target: bool = False) -> SeqBunch:
        """
        :param bunch:
        :param consider_target: should only be set to true in case of training data.
        :return:
        """
        if self._positive_size is None:
            raise 'Call `fit` before calling `transform` because this encoding needs collective sequence information'

        sequences = (bunch.samples['sequence'] if 'sequence' in bunch.samples else bunch.samples[0]).values

        new_samples = []
        for i in range(len(sequences)):
            new_samples.append(
                encode(sequences[i], self.sizes, self.matrices, bunch.targets[i] if consider_target else None))

        return SeqBunch(
            description=bunch.description,
            targets=bunch.targets,
            samples=DataFrame(
                data=new_samples,
                columns=[f'pstnpss_{i}' for i in range(len(self._positives_matrix))]
            )
        )
