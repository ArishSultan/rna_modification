import random


def generate_random_sequence(length, central_nucleotide):
    bases = 'ACGU'
    central_index = length // 2
    random_bases = [random.choice(bases) for _ in range(length)]
    random_bases[central_index] = central_nucleotide
    return ''.join(random_bases)


def subsample_sequence(sequence, min_length=21):
    sequences = [sequence]

    while len(sequence) >= min_length:
        sequence = sequence[1:-1]
        sequences.append(sequence)

    return sequences
