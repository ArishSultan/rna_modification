from pathlib import Path
from pandas import read_csv, DataFrame
from src.utils import get_path, subsample_sequence

_INTERMEDIATE_DIR = get_path() / 'dataset' / 'processed' / 'psi'
_PROCESSED_DIR = get_path() / 'dataset' / 'processed' / 'psi' / 'subsample'


def _process_csv_file(path: Path):
    new_data_0 = []
    new_data_1 = []
    new_data_2 = []

    data = read_csv(path, header=None)
    min_size = 31 if 's.cerevisiae.csv' in str(path) else 21

    for sample in data.iterrows():
        sub_samples = subsample_sequence(sample[1][0], min_size)

        for sub_sample in sub_samples:
            new_data_0.append(sub_sample)
            new_data_1.append(sample[1][1])
            new_data_2.append(sample[1][2])

    new_data = DataFrame({0: new_data_0, 1: new_data_1, 2: new_data_2})

    new_path = _PROCESSED_DIR / path.relative_to(_INTERMEDIATE_DIR)
    new_path.parent.mkdir(exist_ok=True, parents=True)

    new_data.to_csv(new_path, header=False, index=False)


def _main():
    global _augmented_seq

    _augmented_seq = {}

    for i in _INTERMEDIATE_DIR.glob('**/**'):
        for j in i.iterdir():
            if not j.is_file() or not j.name.endswith('.csv'):
                continue

            _process_csv_file(j)


if __name__ == '__main__':
    _main()
