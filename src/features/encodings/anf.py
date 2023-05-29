from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


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
    """
    A transformer that applies the ANF encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the ANF
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the ANF of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import anf
    >>> encoder = anf.Encoder()
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Implementation of base fit_transform.

        Since, there is nothing in `anf` encoding that needs fitting so, it just
        transforms all the sequences to their `anf` encoding.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return x.applymap(encode)

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of ANF-encoded sequences.
        """
        return self.fit_transform(x)
