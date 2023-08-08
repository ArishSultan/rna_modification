from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin

from ...data.seq_bunch import SeqBunch
from ...utils.features import encode_df


def encode(sequence: str) -> list[str]:
    mol = Chem.MolFromSequence(sequence, flavor=5)
    return list(str(Chem.MolToInchi(mol))) if mol else list()


class Encoder(BaseEstimator, TransformerMixin):
    def fit_transform(self, bunch: SeqBunch, **kwargs) -> SeqBunch:
        return SeqBunch(
            targets=bunch.targets,
            description=bunch.description,
            samples=encode_df(bunch.samples, lambda seq: encode(seq), 'inchi')
        )

    def transform(self, x: SeqBunch) -> SeqBunch:
        """
        Just a wrapper around `fit_transform` that calls the base method.

        :param x: A DataFrame of DNA/RNA sequences.
        :return: A DataFrame of Kmer-encoded sequences.
        """
        return self.fit_transform(x)
