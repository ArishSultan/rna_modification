from pandas import concat

from filter_dataset import save_dataset, similarity_report, separate_unique_sequences, human_dataset, yeast_dataset, \
    mouse_dataset, DATASET_PATH
from filter_dataset_indepent import human_benchmark_independent_dataset, yeast_benchmark_independent_dataset
from filter_dataset_training import human_benchmark_training_dataset, mouse_benchmark_training_dataset, \
    yeast_benchmark_training_dataset


def _main():
    human_benchmark_dataset = concat([
        human_benchmark_training_dataset,
        human_benchmark_independent_dataset
    ])

    mouse_benchmark_dataset = mouse_benchmark_training_dataset

    yeast_benchmark_dataset = concat([
        yeast_benchmark_training_dataset,
        yeast_benchmark_independent_dataset
    ])

    human_benchmark_dataset.reset_index(inplace=True, drop=True)
    mouse_benchmark_dataset.reset_index(inplace=True, drop=True)
    yeast_benchmark_dataset.reset_index(inplace=True, drop=True)

    human_dataset_report = similarity_report(human_dataset, human_benchmark_dataset)
    human_filtered_dataset = separate_unique_sequences(human_dataset, human_dataset_report)

    yeast_dataset_report = similarity_report(yeast_dataset, yeast_benchmark_dataset, 5)
    yeast_filtered_dataset = separate_unique_sequences(yeast_dataset, yeast_dataset_report)

    mouse_dataset_report = similarity_report(mouse_dataset, mouse_benchmark_dataset)
    mouse_filtered_dataset = separate_unique_sequences(mouse_dataset, mouse_dataset_report)

    directory = DATASET_PATH / 'intermediate' / 'psi' / 'filtered' / 'all'
    directory.mkdir(parents=True, exist_ok=True)

    save_dataset(directory / 'h.sapiens.csv', human_filtered_dataset)
    save_dataset(directory / 'm.musculus.csv', mouse_filtered_dataset)
    save_dataset(directory / 's.cerevisiae.csv', yeast_filtered_dataset)


if __name__ == '__main__':
    _main()
