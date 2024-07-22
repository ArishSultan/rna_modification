import os

from pandas import read_csv, concat
from sklearn.model_selection import train_test_split

from src.utils import get_path

from . import SeqBunch, Species, Modification


def load_benchmark_dataset(species: Species, modification: Modification, independent: bool = False) -> SeqBunch | None:
    root_dir = get_path()
    group = 'independent' if independent else 'training'
    file_path = root_dir / 'dataset' / 'benchmark' / modification.value / group / f'{species.value}.csv'

    if not os.path.exists(file_path):
        return None

    data = read_csv(file_path, header=None)

    return SeqBunch(
        targets=data[1],
        samples=data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
    )


def load_dataset(species: Species, modification: Modification, filtered=None, subsample=False,
                 seq_len: int = 0) -> SeqBunch:
    root_dir = get_path()
    file_path = root_dir / 'dataset' / 'processed' / modification.value

    if subsample:
        file_path = file_path / 'subsample'

    if filtered is not None:
        file_path = file_path / 'filtered' / filtered

    file_path = file_path / f'{species.value}.csv'

    data = read_csv(file_path, header=None)

    if seq_len > 0:
        data = data[data[0].apply(len) == 21]
        data.reset_index(drop=True, inplace=True)

    if 2 in data:
        return SeqBunch(
            targets=data[1],
            samples=data.drop([1, 2], axis=1).rename({0: 'sequence'}, axis=1),
        )
    else:
        return SeqBunch(
            targets=data[1],
            samples=data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
        )


def split_balanced(x, y, test_size=0.2):
    p = y == 1
    n = y == 0

    p_x, p_y = x[p], y[p]
    n_x, n_y = x[n], y[n]

    train_p_x, test_p_x, train_p_y, test_p_y = train_test_split(p_x, p_y, test_size=test_size, random_state=42)
    train_n_x, test_n_x, train_n_y, test_n_y = train_test_split(n_x, n_y, test_size=test_size, random_state=42)

    return (concat([train_p_x, train_n_x]), concat([test_p_x, test_n_x]),
            concat([train_p_y, train_n_y]), concat([test_p_y, test_n_y]))
