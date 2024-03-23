# from rdkit import Chem
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.features import encode_df


def encode(sequence: str) -> list[str]:
    pass
    # mol = Chem.MolFromSequence(sequence, flavor=5)
    # return list(str(Chem.MolToInchi(mol))) if mol else list()


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, encode, 'inchi')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
