from . import knc


def encode(sequence: str) -> list[float]:
    return knc.encode(sequence, 3)


class Encoder(knc.Encoder):
    def __init__(self):
        super().__init__(3)
