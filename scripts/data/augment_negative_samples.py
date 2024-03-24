from pathlib import Path
from itertools import chain
from pandas import read_csv, concat, DataFrame
from src.utils import get_path, generate_random_sequence

_augmented_seq = {}

_INTERMEDIATE_DIR = get_path() / 'dataset' / 'intermediate' / 'psi'
_PROCESSED_DIR = get_path() / 'dataset' / 'processed' / 'psi'


def _augment_seq(sample: str, all_samples: list[str]):
    global _augmented_seq

    if sample in _augmented_seq:
        return _augmented_seq[sample]

    new_sample = generate_random_sequence(len(sample), 'U')
    while new_sample in all_samples:
        new_sample = generate_random_sequence(len(sample), 'U')

    _augmented_seq[sample] = new_sample

    return new_sample


def _process_csv_file(path: Path):
    new_labels = []
    new_seqs = []

    data = read_csv(path, header=None)
    sequences = data[0].values

    for sample in sequences:
        new_seq = _augment_seq(sample, list(chain(sequences, new_seqs)))
        new_seqs.append(new_seq)
        new_labels.append(0)

    new_data = concat([
        data,
        DataFrame({0: new_seqs, 1: new_labels, 2: 1})
    ])

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
