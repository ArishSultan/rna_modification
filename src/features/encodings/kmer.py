from pandas import DataFrame, Series
from itertools import product
from collections import Counter

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def kmer_count(sequence: str, k: int = 2, upto: bool = False, normalize: bool = False) -> dict:
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


def seq_to_kmer(sequence: str, k: int = 2, upto: bool = False) -> list[str]:
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

    return results


def generate_all_kmers(k: int = 2, upto: bool = False) -> list[str]:
    assert k > 0
    nucleotides = 'ACGU'

    if upto:
        repeats = list(range(1, k + 1))
    else:
        repeats = [k]

    results = list[str]()
    for i in repeats:
        results += [''.join(x) for x in product(nucleotides, repeat=i)]

    return results


def encode(sequence: str, k: int = 2, upto: bool = False, normalize: bool = False) -> list[float]:
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


class Encoder(BaseEncoder):
    def __init__(self, k: int = 2, upto: bool = False, normalize: bool = False):
        self.k = k
        self.upto = upto
        self.normalize = normalize

    def fit(self, x: DataFrame, y: Series):
        print('KMER encoding is unsupervised and does not need to be fitted on training data')

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.k, self.upto, self.normalize), 'kmer')

    def transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return self.fit_transform(x)
