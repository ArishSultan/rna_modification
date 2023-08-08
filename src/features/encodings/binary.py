from sklearn.base import BaseEstimator, TransformerMixin

from ...data.seq_bunch import SeqBunch
from ...utils.features import encode_df


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
    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:
        return SeqBunch(
            targets=bunch.targets,
            description=bunch.description,
            samples=encode_df(bunch.samples, encode, 'binary'),
        )

    def transform(self, x: SeqBunch) -> SeqBunch:
        return self.fit_transform(x)
