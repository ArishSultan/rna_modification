from pathlib import Path
from pandas import DataFrame, read_csv
from src.utils import get_path

DATASET_PATH = get_path() / 'dataset'

human_dataset = read_csv(DATASET_PATH / 'intermediate' / 'psi' / 'h.sapiens.csv', header=None)
mouse_dataset = read_csv(DATASET_PATH / 'intermediate' / 'psi' / 'm.musculus.csv', header=None)
yeast_dataset = read_csv(DATASET_PATH / 'intermediate' / 'psi' / 's.cerevisiae.csv', header=None)


def similarity_report(dataset: DataFrame, benchmark_dataset: DataFrame, diff=10):
    report = {}

    sequences = dataset[0]
    benchmark_sequences = benchmark_dataset[0]

    for i in range(len(sequences)):
        seq = sequences[i][diff: -diff]

        for j in range(len(benchmark_sequences)):
            if seq == benchmark_sequences[j]:
                if i in report:
                    report[i].append(j)
                else:
                    report[i] = [j]

    return report


def separate_unique_sequences(dataset: DataFrame, indexes: dict):
    new_dataset = dataset
    for index in indexes.keys():
        new_dataset = new_dataset.drop(index, axis=0)

    return new_dataset.reset_index(drop=True)


def save_dataset(directory: Path, dataset: DataFrame):
    dataset.to_csv(directory, index=False, header=False)
