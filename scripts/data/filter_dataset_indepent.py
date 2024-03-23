from pandas import read_csv
from filter_dataset import save_dataset, similarity_report, separate_unique_sequences, human_dataset, yeast_dataset, \
    DATASET_PATH

human_benchmark_independent_dataset = read_csv(
    DATASET_PATH / 'benchmark' / 'psi' / 'independent' / 'h.sapiens.csv',
    header=None
)

yeast_benchmark_independent_dataset = read_csv(
    DATASET_PATH / 'benchmark' / 'psi' / 'independent' / 's.cerevisiae.csv',
    header=None
)


def _main():
    human_dataset_report = similarity_report(human_dataset, human_benchmark_independent_dataset)
    human_filtered_dataset = separate_unique_sequences(human_dataset, human_dataset_report)

    yeast_dataset_report = similarity_report(yeast_dataset, yeast_benchmark_independent_dataset, 5)
    yeast_filtered_dataset = separate_unique_sequences(yeast_dataset, yeast_dataset_report)

    directory = DATASET_PATH / 'filtered' / 'psi' / 'independent'
    directory.mkdir(parents=True, exist_ok=True)

    save_dataset(directory / 'h.sapiens.csv', human_filtered_dataset)
    save_dataset(directory / 's.cerevisiae.csv', yeast_filtered_dataset)


if __name__ == '__main__':
    _main()
