from pandas import read_csv
from filter_dataset import similarity_report, separate_unique_sequences, save_dataset, human_dataset, yeast_dataset, \
    mouse_dataset, DATASET_PATH

human_benchmark_training_dataset = read_csv(
    DATASET_PATH / 'benchmark' / 'psi' / 'training' / 'h.sapiens.csv',
    header=None
)

mouse_benchmark_training_dataset = read_csv(
    DATASET_PATH / 'benchmark' / 'psi' / 'training' / 'm.musculus.csv',
    header=None
)

yeast_benchmark_training_dataset = read_csv(
    DATASET_PATH / 'benchmark' / 'psi' / 'training' / 's.cerevisiae.csv',
    header=None
)


def _main():
    human_dataset_report = similarity_report(human_dataset, human_benchmark_training_dataset)
    human_filtered_dataset = separate_unique_sequences(human_dataset, human_dataset_report)

    mouse_dataset_report = similarity_report(mouse_dataset, mouse_benchmark_training_dataset)
    mouse_filtered_dataset = separate_unique_sequences(mouse_dataset, mouse_dataset_report)

    yeast_dataset_report = similarity_report(yeast_dataset, yeast_benchmark_training_dataset, 5)
    yeast_filtered_dataset = separate_unique_sequences(yeast_dataset, yeast_dataset_report)

    directory = DATASET_PATH / 'filtered' / 'psi' / 'training'
    directory.mkdir(parents=True, exist_ok=True)

    save_dataset(directory / 'h.sapiens.csv', human_filtered_dataset)
    save_dataset(directory / 'm.musculus.csv', mouse_filtered_dataset)
    save_dataset(directory / 's.cerevisiae.csv', yeast_filtered_dataset)


if __name__ == '__main__':
    _main()
