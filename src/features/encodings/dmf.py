import numpy as np
from collections import Counter
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer as kmer_enc


def _calculate_mi(pos_counts, neg_counts, pos_total, neg_total, total):
    mi = 0
    for kmer in set(pos_counts.keys()) | set(neg_counts.keys()):
        pos_freq = pos_counts.get(kmer, 0) / pos_total
        neg_freq = neg_counts.get(kmer, 0) / neg_total
        total_freq = (pos_counts.get(kmer, 0) + neg_counts.get(kmer, 0)) / total

        if pos_freq > 0:
            mi += pos_freq * np.log2(pos_freq / total_freq)
        if neg_freq > 0:
            mi += neg_freq * np.log2(neg_freq / total_freq)

    return mi


def _count_kmers(sequences, k):
    kmer_counts = Counter()
    for seq in sequences:
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        kmer_counts.update(kmers)
    return kmer_counts


def encode(sequence: str, top_motifs: list) -> list:
    return [1 if motif in sequence else 0 for motif in top_motifs]


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, top_n=10):
        self.k = k
        self.top_n = top_n
        self.top_motifs = ['AAA', 'ACC', 'AUA', 'CGA', 'CGC', 'CGG', 'GCG', 'GGC', 'GUA', 'UCG']

    def fit(self, x: DataFrame, y: Series):
        positives = x[y == 1]['sequence'].values
        negatives = x[y == 0]['sequence'].values

        pos_counts = _count_kmers(positives, self.k)
        neg_counts = _count_kmers(negatives, self.k)

        pos_total = sum(pos_counts.values())
        neg_total = sum(neg_counts.values())
        total = pos_total + neg_total

        mi_scores = {}
        for kmer in kmer_enc.generate_all_kmers(self.k):
            mi = _calculate_mi({kmer: pos_counts.get(kmer, 0)},
                               {kmer: neg_counts.get(kmer, 0)},
                               pos_total, neg_total, total)
            mi_scores[kmer] = mi

        self.top_motifs = sorted(mi_scores, key=mi_scores.get, reverse=True)[:self.top_n]

    def transform(self, x: DataFrame) -> DataFrame:
        if self.top_motifs is None:
            raise ValueError(
                'Call `fit` before calling `transform` because this encoding needs collective sequence information')

        sequences = (x['sequence'] if 'sequence' in x else x[0]).values

        encoded_samples = [encode(seq, self.top_motifs) for seq in sequences]

        return DataFrame(
            data=encoded_samples,
            columns=[f'dmf_{i}' for i in range(self.top_n)]
        )

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        self.fit(x, kwargs['y'])
        return self.transform(x)
