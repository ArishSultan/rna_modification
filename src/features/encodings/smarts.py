from rdkit import Chem
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import encode_df


def encode(sequence: str) -> list[str]:
    mol = Chem.MolFromSequence(sequence, flavor=5)
    return list(str(Chem.MolToSmarts(mol))) if mol else list()


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq), 'inchi')

    def transform(self, x: DataFrame) -> DataFrame:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Kmer-encoded sequences.
        """
        return self.fit_transform(x)
