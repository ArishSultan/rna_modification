from pathlib import Path


def _generate_visualization_grid(train_samples, train_targets, test_samples, test_targets, folder_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from umap import UMAP
    from sklearn.manifold import TSNE
    from src.features.encodings import binary, nd, ncp, kmer, pse_knc, pstnpss

    encoders = [
        binary.Encoder(),
        nd.Encoder(),
        ncp.Encoder(),
        kmer.Encoder(k=3),
        pse_knc.Encoder(),
        pstnpss.Encoder(),
    ]

    reducers = [
        (TSNE(n_components=2), False),
        (UMAP(n_components=2), False),
        (UMAP(n_components=2), True),
    ]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, encoder in enumerate(encoders):
        encoded_train_samples = encoder.fit_transform(train_samples, y=train_targets)
        if test_samples is not None:
            encoded_test_samples = encoder.transform(test_samples)
        else:
            encoded_test_samples = None

        for j, item in enumerate(reducers):
            reducer, flag = item
            if not flag and encoded_test_samples is not None:
                combined_samples = np.vstack((encoded_train_samples, encoded_test_samples))
                reduced_combined_samples = reducer.fit_transform(combined_samples)

                reduced_train_samples = reduced_combined_samples[:len(train_samples)]
                reduced_test_samples = reduced_combined_samples[len(train_samples):]
            else:
                if flag:
                    reduced_train_samples = reducer.fit_transform(encoded_train_samples, y=train_targets)
                    reduced_test_samples = reducer.transform(
                        encoded_test_samples) if encoded_test_samples is not None else None
                else:
                    reduced_train_samples = reducer.fit_transform(encoded_train_samples)
                    reduced_test_samples = reducer.fit_transform(
                        encoded_test_samples) if encoded_test_samples is not None else None

            # Plot the reduced samples
            plt.figure(figsize=(6, 6))
            scatter_test = None
            scatter_train_positive = plt.scatter(reduced_train_samples[train_targets == 1, 0],
                                                 reduced_train_samples[train_targets == 1, 1],
                                                 c='green', label='Train +ive', marker='o', alpha=0.7)
            scatter_train_negative = plt.scatter(reduced_train_samples[train_targets == 0, 0],
                                                 reduced_train_samples[train_targets == 0, 1],
                                                 c='blue', label='Train -ive', marker='o', alpha=0.7)

            if reduced_test_samples is not None and test_targets is not None:
                scatter_test_positive = plt.scatter(reduced_test_samples[test_targets == 1, 0],
                                                    reduced_test_samples[test_targets == 1, 1],
                                                    c='red', label='Test +ive', marker='x', alpha=0.7)
                scatter_test_negative = plt.scatter(reduced_test_samples[test_targets == 0, 0],
                                                    reduced_test_samples[test_targets == 0, 1],
                                                    c='purple', label='Test -ive', marker='x', alpha=0.7)

            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()

            plt.savefig(folder_name / f'plot_{i + 1}_{j + 1}.png')
            plt.close()


if __name__ == '__main__':
    from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch

    human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
    human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)

    mouse_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

    yeast_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
    yeast_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)

    dir = Path('/Users/arish/Research/research/rna_modification/writing/chapter/images/encodings/unsupervised')
    dir.mkdir(exist_ok=True, parents=True)

    _generate_visualization_grid(human_train_dataset.samples, human_train_dataset.targets.values,
                                 human_test_dataset.samples,
                                 human_test_dataset.targets.values, dir / 'human')

    _generate_visualization_grid(mouse_train_dataset.samples, mouse_train_dataset.targets.values,
                                 None,
                                 None, dir / 'mouse')

    _generate_visualization_grid(yeast_train_dataset.samples, yeast_train_dataset.targets.values,
                                 yeast_test_dataset.samples,
                                 yeast_test_dataset.targets.values, dir / 'yeast')
