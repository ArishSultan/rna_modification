from pandas import read_csv
from src.utils import get_path

from . import SeqBunch, Species, Modification


def load_benchmark_dataset(species: Species, modification: Modification, independent: bool = False) -> SeqBunch:
    root_dir = get_path()
    group = 'independent' if independent else 'training'
    file_path = root_dir / 'dataset' / 'benchmark' / modification.value / group / f'{species.value}.csv'

    data = read_csv(file_path, header=None)

    return SeqBunch(
        targets=data[1],
        samples=data[0],
    )


def load_dataset(species: Species, modification: Modification, filtered=None, subsample=False) -> SeqBunch:
    root_dir = get_path()
    file_path = root_dir / 'dataset' / 'processed' / modification.value

    if subsample:
        file_path = file_path / 'subsample'

    if filtered is not None:
        file_path = file_path / 'filtered' / filtered

    file_path = file_path / f'{species.value}.csv'

    data = read_csv(file_path, header=None)

    return SeqBunch(
        targets=data[1],
        samples=data[0],
    )
