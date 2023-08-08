from . import knc


def encode(sequence: str) -> list[float]:
    """
    A derivation of KNC in which k is fixed to 2.

    Parameters:
    sequence (str): The input DNA/RNA sequence (S).

    Returns:
    list[float]: A list representing the normalized count (frequency) of each 2-tuple in the sequence.
    """
    return knc.encode(sequence, 2)


class Encoder(knc.Encoder):
    def __init__(self):
        super().__init__(3)
