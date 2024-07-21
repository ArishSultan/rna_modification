from collections import defaultdict

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

from . import pstnpss

def calculate_trinucleotide_frequencies(sequences, l, xi):
    forward_frequencies = defaultdict(lambda: np.zeros(64))
    backward_frequencies = defaultdict(lambda: np.zeros(64))
    trinucleotides = ["".join([x, y, z]) for x in "ACGU" for y in "ACGU" for z in "ACGU"]
    trinucleotide_index = {trinucleotide: idx for idx, trinucleotide in enumerate(trinucleotides)}

    for seq in sequences:
        for i in range(l - xi - 2):
            forward_trinucleotide = seq[i] + seq[i + xi + 1] + seq[i + xi + 2]
            backward_trinucleotide = seq[i + xi + 2] + seq[i + xi + 1] + seq[i]

            forward_frequencies[i][trinucleotide_index[forward_trinucleotide]] += 1
            backward_frequencies[i + xi + 2][trinucleotide_index[backward_trinucleotide]] += 1

    for i in range(l - xi - 2):
        total_forward = sum(forward_frequencies[i])
        total_backward = sum(backward_frequencies[i + xi + 2])

        if total_forward > 0:
            forward_frequencies[i] /= total_forward
        if total_backward > 0:
            backward_frequencies[i + xi + 2] /= total_backward

    return forward_frequencies, backward_frequencies


def encode_rna_sequence(seq, forward_frequencies_pos, backward_frequencies_pos, forward_frequencies_neg,
                        backward_frequencies_neg, l, xi):
    encoded_vector = []
    for i in range(xi + 3, l - xi - 2):
        forward_pos = forward_frequencies_pos[i][pstnpss.TRI_NUCLEOTIDES_DICT[seq[i] + seq[i + xi + 1] + seq[i + xi + 2]]]
        backward_pos = backward_frequencies_pos[i][pstnpss.TRI_NUCLEOTIDES_DICT[seq[i + xi + 2] + seq[i + xi + 1] + seq[i]]]

        forward_neg = forward_frequencies_neg[i][pstnpss.TRI_NUCLEOTIDES_DICT[seq[i] + seq[i + xi + 1] + seq[i + xi + 2]]]
        backward_neg = backward_frequencies_neg[i][pstnpss.TRI_NUCLEOTIDES_DICT[seq[i + xi + 2] + seq[i + xi + 1] + seq[i]]]

        v_pos = (forward_pos + backward_pos) / 2
        v_neg = (forward_neg + backward_neg) / 2

        encoded_value = v_pos - v_neg
        encoded_vector.append(encoded_value)

    return encoded_vector


# Example datasets
# D_plus = ["ACGUAGCUAGCUA", "CGUAUGCUAGCUA"]  # Positive dataset
# D_minus = ["UAGCUAGCUAGCU", "AGCUAGCUAGCUA"]  # Negative dataset
# l = len(D_plus[0])
# xi_values = range(0, (l - 5) // 2 + 1)
#
# # Calculate frequencies for different xi values
# forward_frequencies_pos_all = []
# backward_frequencies_pos_all = []
# forward_frequencies_neg_all = []
# backward_frequencies_neg_all = []
#
# for xi in xi_values:
#     forward_frequencies_pos, backward_frequencies_pos = calculate_trinucleotide_frequencies(D_plus, l, xi)
#     forward_frequencies_neg, backward_frequencies_neg = calculate_trinucleotide_frequencies(D_minus, l, xi)
#
#     forward_frequencies_pos_all.append(forward_frequencies_pos)
#     backward_frequencies_pos_all.append(backward_frequencies_pos)
#     forward_frequencies_neg_all.append(forward_frequencies_neg)
#     backward_frequencies_neg_all.append(backward_frequencies_neg)
#
# # Encode sequences
# encoded_vectors = []
# for seq in D_plus:
#     encoded_vector = []
#     for xi in xi_values:
#         encoded_vector.extend(encode_rna_sequence(seq, forward_frequencies_pos_all[xi], backward_frequencies_pos_all[xi], forward_frequencies_neg_all[xi], backward_frequencies_neg_all[xi], l, xi))
#     encoded_vectors.append(encoded_vector)
#
# for seq in D_minus:
#     encoded_vector = []
#     for xi in xi_values:
#         encoded_vector.extend(encode_rna_sequence(seq, forward_frequencies_pos_all[xi], backward_frequencies_pos_all[xi], forward_frequencies_neg_all[xi], backward_frequencies_neg_all[xi], l, xi))
#     encoded_vectors.append(encoded_vector)
#
# # Output the encoded vectors
# for vec in encoded_vectors:
#     print(vec)


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, xi):
        self.xi = xi
        self.forward_frequencies_pos_all = []
        self.forward_frequencies_neg_all = []
        self.backward_frequencies_pos_all = []
        self.backward_frequencies_neg_all = []

    def fit(self, x: DataFrame, y: Series):
        self.forward_frequencies_pos_all = []
        self.forward_frequencies_neg_all = []
        self.backward_frequencies_pos_all = []
        self.backward_frequencies_neg_all = []

        D_plus = x[y == 1]['sequence'].values
        D_minus = x[y == 0]['sequence'].values

        for xi in self.xi:
            l = len(D_plus[0])
            forward_frequencies_pos, backward_frequencies_pos = calculate_trinucleotide_frequencies(D_plus, l, xi)
            forward_frequencies_neg, backward_frequencies_neg = calculate_trinucleotide_frequencies(D_minus, l, xi)

            self.forward_frequencies_pos_all.append(forward_frequencies_pos)
            self.backward_frequencies_pos_all.append(backward_frequencies_pos)
            self.forward_frequencies_neg_all.append(forward_frequencies_neg)
            self.backward_frequencies_neg_all.append(backward_frequencies_neg)

    def fit_transform(self, x: DataFrame, y: Series, **kwargs) -> DataFrame:
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x: DataFrame) -> DataFrame:
        sequences = (x['sequence'] if 'sequence' in x else x[0]).values

        new_sample = []
        for seq in sequences:
            encoded_vector = []
            for xi in self.xi:
                encoded_vector.extend(
                    encode_rna_sequence(seq, self.forward_frequencies_pos_all[xi],
                                        self.backward_frequencies_pos_all[xi],
                                        self.forward_frequencies_neg_all[xi], self.backward_frequencies_neg_all[xi],
                                        len(seq),
                                        xi))

            new_sample.append(encoded_vector)

        return DataFrame(
            data=new_sample,
            columns=[f'pstnpss_{i}' for i in range(len(new_sample[0]))]
        )
