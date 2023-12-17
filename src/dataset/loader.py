from pandas import read_csv
from src.utils import get_path

from . import SeqBunch, Species, Modification


def load_benchmark_dataset(species: Species, modification: Modification, independent: bool = False) -> SeqBunch:
    root_dir = get_path()
    group = 'independent' if independent else 'training'
    file_path = root_dir / 'data' / 'benchmark' / modification.value / group / f'{species.value}.csv'

    data = read_csv(file_path, header=None)

    return SeqBunch(
        targets=data[1],
        samples=data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
    )


def load_dataset(species: Species, modification: Modification) -> SeqBunch:
    root_dir = get_path()
    file_path = root_dir / 'data' / 'processed' / modification.value / f'{species.value}.csv'

    data = read_csv(file_path, header=None)

    return SeqBunch(
        targets=data[1],
        samples=data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
    )
