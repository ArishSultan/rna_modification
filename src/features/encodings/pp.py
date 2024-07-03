import json
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import get_path
from ...utils.features import encode_df

with open(get_path("src/features/encodings/1_positional_probabilities.json")) as file:
    positional_probabilities = json.load(file)


def encode(sequence: str) -> list[tuple[float, float]]:
    result = []

    mid = len(sequence) // 2
    sequence = sequence[:mid] + sequence[mid + 1:]

    for i, nucleotide in enumerate(sequence):
        pos_probs = positional_probabilities[i]["pos"]
        neg_probs = positional_probabilities[i]["neg"]

        result.append(pos_probs[nucleotide] - neg_probs[nucleotide])

    return result


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'pp')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
