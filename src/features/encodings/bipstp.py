import numpy as np
from collections import defaultdict, Counter


def calculate_trinucleotide_frequencies(sequences, k, gap):
    forward_frequencies = defaultdict(lambda: np.zeros(64))
    backward_frequencies = defaultdict(lambda: np.zeros(64))
    trinucleotides = ["".join([x, y, z]) for x in "ACGU" for y in "ACGU" for z in "ACGU"]
    trinucleotide_index = {trinucleotide: idx for idx, trinucleotide in enumerate(trinucleotides)}

    for seq in sequences:
        for i in range(l - gap - k):
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
        forward_pos = forward_frequencies_pos[i][trinucleotide_index[seq[i] + seq[i + xi + 1] + seq[i + xi + 2]]]
        backward_pos = backward_frequencies_pos[i][trinucleotide_index[seq[i + xi + 2] + seq[i + xi + 1] + seq[i]]]

        forward_neg = forward_frequencies_neg[i][trinucleotide_index[seq[i] + seq[i + xi + 1] + seq[i + xi + 2]]]
        backward_neg = backward_frequencies_neg[i][trinucleotide_index[seq[i + xi + 2] + seq[i + xi + 1] + seq[i]]]

        v_pos = (forward_pos + backward_pos) / 2
        v_neg = (forward_neg + backward_neg) / 2

        encoded_value = v_pos - v_neg
        encoded_vector.append(encoded_value)

    return encoded_vector

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
#
