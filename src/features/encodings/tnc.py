from . import knc


def encode(sequence: str) -> list[float]:
    """
    A derivation of KNC in which k is fixed to 3.

    Parameters:
    sequence (str): The input DNA/RNA sequence (S).

    Returns:
    list[float]: A list representing the normalized count (frequency) of each 3-tuple in the sequence.
    """
    return knc.encode(sequence, 3)


class Encoder(knc.Encoder):
    """
    A transformer that applies the TNC encoding technique to DNA/RNA sequences.

    This transformer takes a DataFrame of DNA/RNA sequences and applies the TNC
    encoding technique to each sequence. The resulting DataFrame contains a list
    of floats representing the TNC of each sequence.

    Example usage:
    >>> from pandas import DataFrame
    >>> from src.features import tnc
    >>> encoder = tnc.Encoder()
    >>> sequences = DataFrame(["CAUGGAG", "ACGTACGTACGT"])
    >>> encoded_sequences = encoder.fit_transform(sequences)
    """

    def __init__(self):
        super().__init__(3)
