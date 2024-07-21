from collections import Counter
from pandas import DataFrame, Series

from ..encoder import BaseEncoder
from ...utils.features import encode_df


def _resolve_value(motif, nucleotide, proba):
    return proba[motif][nucleotide] / sum(proba[motif].values(), 0) if motif in proba else 0


def encode(sequence: str, pos_prob, neg_prob):
    mid = len(sequence) // 2
    pos_result = [0] * mid * 2
    neg_result = [0] * mid * 2

    for i in range(mid):
        l_limit = mid - i - 1
        r_limit = mid + i + 2

        l_seq = sequence[l_limit: mid][1:]
        r_seq = sequence[mid + 1: r_limit][:-1]

        pos_result[l_limit] = _resolve_value(l_seq, sequence[l_limit], pos_prob[0])
        pos_result[r_limit - 2] = _resolve_value(r_seq, sequence[r_limit - 1], pos_prob[1])

        neg_result[l_limit] = _resolve_value(l_seq, sequence[l_limit], neg_prob[0])
        neg_result[r_limit - 2] = _resolve_value(r_seq, sequence[r_limit - 1], neg_prob[1])

    return pos_result + neg_result


def _process_one_side(start, end, samples, proba, reverse=False):
    uniques = set()
    for sample in samples:
        uniques.add(sample[start: end])

    index = start - 1 if reverse else end

    if end == start:
        nucleotides = []
        for sample in samples:
            nucleotides.append(sample[index])
        proba[''] = Counter(nucleotides)
        return

    for item in uniques:
        nucleotides = []
        for sample in samples:
            if sample[start: end] == item:
                nucleotides.append(sample[index])
        proba[item] = Counter(nucleotides)


def _prepare_probabilities(samples):
    if len(samples) == 0:
        return None

    r_prob = dict()
    l_prob = dict()
    mid = len(samples[0]) // 2

    for i in range(mid):
        l_limit = mid - i
        r_limit = mid + i + 1

        _process_one_side(mid + 1, r_limit, samples, r_prob)
        _process_one_side(l_limit, mid, samples, l_prob, True)

    return r_prob, l_prob


class Encoder(BaseEncoder):
    def __init__(self):
        self.pos_prob = None
        self.neg_prob = None

    def fit(self, x: DataFrame, y: Series):
        positive_samples = x[y == 1]['sequence'].values
        negative_samples = x[y == 0]['sequence'].values

        self.pos_prob = _prepare_probabilities(positive_samples)
        self.neg_prob = _prepare_probabilities(negative_samples)

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        self.fit(x, **kwargs)
        return self.transform(x)

    def transform(self, x: DataFrame) -> DataFrame:
        return encode_df(x, lambda x: encode(x, self.pos_prob, self.neg_prob), 'proba')
