from sklearn.base import BaseEstimator, TransformerMixin

from ...data.seq_bunch import SeqBunch
from ...utils.features import encode_df


def encode(sequence: str) -> list[float]:
    """
    Accumulated Nucleotide Frequency (ANF) is a feature encoding technique used
    in bioinformatics to represent *DNA/RNA* sequences. It calculates the frequency
    of each nucleotide (A, C, G, T/U) at each position in a *DNA/RNA* sequence,
    and then accumulates these frequencies over the length of the sequence.
    The resulting feature vector represents the accumulated frequency of each
    nucleotide at each position in the sequence.

    **ANF** is useful in bioinformatics because it can capture important information
    about the structure and function of *DNA/RNA* sequences. For example, **ANF**
    has been used to predict *protein-DNA* binding sites, classify *DNA* sequences
    by their function, and identify regions of *DNA* that are conserved across
    different species.

    :param sequence: A string representing a DNA/RNA sequence.
    :return: A list of floats representing the ANF of the input sequence.

    Example:
    >>> encode("CAUGGAGAGAUGUUCUUUACU")
    """
    return list(
        sequence[0: j + 1].count(sequence[j]) / (j + 1)
        for j in range(len(sequence))
    )


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:

        return SeqBunch(
            targets=bunch.targets,
            description=bunch.description,
            samples=encode_df(bunch.samples, encode, 'anf')
        )

    def transform(self, x: SeqBunch) -> SeqBunch:
        return self.fit_transform(x)
