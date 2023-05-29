from pandas import read_csv

from ..utils import get_path
from .species import Species
from .seq_bunch import SeqBunch


def _resolve_dataset(name: str, independent: bool, species: Species):
    group = ('independent' if independent else 'training')
    return get_path(f'data') / 'raw' / name / group / f'{species.value}.csv'


def load_psi(species: Species, independent: bool = False) -> SeqBunch:
    dataset_path = _resolve_dataset('psi', independent, species)
    dataset = read_csv(dataset_path, header=None)

    return SeqBunch(
        targets=dataset[1],
        samples=dataset.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
        description=f'Pseudo-uridine modification dataset for {species.value}',
    )