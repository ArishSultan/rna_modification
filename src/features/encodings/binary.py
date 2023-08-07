from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import encode_df


def encode(sequence: str) -> list[float]:
    """
    Encode a *DNA/RNA* sequence using binary feature encoding.

    **Binary** feature encoding is a method used to represent *DNA/RNA* sequences
    as *binary vectors*, with each position in the vector corresponding to a specific
    feature of the sequence. In this method, each nucleotide base (A, C, G, T/U)
    is assigned a binary code, such as 00, 01, 10, and 11. Each position in the
    binary vector corresponds to a specific nucleotide base at a specific position
    in the *DNA/RNA* sequence, and the value at that position is the binary code
    for that nucleotide.

    Mathematically, binary feature encoding can be represented using a one-hot
    encoding scheme, where each nucleotide is represented by a vector of zeros
    with a single one at the corresponding position. The resulting binary vector
    is the concatenation of the one-hot vectors for each nucleotide in the sequence.

    Let $S$ be a *DNA/RNA* sequence of length $n$, where $S_i$ is the nucleotide
    at position $i$ in the sequence. Let $C(S_i)$ be the binary code for nucleotide
    $S_i$, such that,

    $$C(S_i) = (c_{i,1}, c_{i,2}, c_{i,3}, c_{i,4})$$ $$\text{where $c_{i,j}$ is 
    $1$ if $S_i$ is nucleotide $j$ and $0$ otherwise}$$

    Then the binary feature encoding of $S$ is given by the concatenation of the
    binary codes for each nucleotide in $S$: $$encode(S) = (c_{1,1}, c_{1,2},
    c_{1,3}, c_{1,4}, c_{2,1}, c_{2,2}, c_{2,3}, c_{2,4}, ..., c_{n,1}, c_{n,2},
    c_{n,3}, c_{n,4})$$

    :param sequence: DNA/RNA sequence to be encoded
    :return: Binary encoded sequence as a tuple of floats
    """
    nucleotide_codes = {'A': [1.0, 0.0, 0.0, 0.0],
                        'C': [0.0, 1.0, 0.0, 0.0],
                        'G': [0.0, 0.0, 1.0, 0.0],
                        'T': [0.0, 0.0, 0.0, 1.0],
                        'U': [0.0, 0.0, 0.0, 1.0]}

    encoded_seq = []

    for nucleotide in sequence:
        encoded_seq += nucleotide_codes.get(nucleotide, [0.0, 0.0, 0.0, 0.0])

    return encoded_seq


class Encoder(BaseEstimator, TransformerMixin):
    """
    A transformer that applies the Binary encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the Binary
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the Binary of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import binary
    >>> encoder = binary.Encoder()
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Implementation of base fit_transform.

        Since, there is nothing in `binary` encoding that needs fitting so, it just
        transforms all the sequences to their `binary` encoding.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Binary-encoded sequences.
        """
        return encode_df(x, encode, 'binary')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Binary-encoded sequences.
        """
        return self.fit_transform(x)
