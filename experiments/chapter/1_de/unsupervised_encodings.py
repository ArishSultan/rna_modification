def _generate_visualization_grid(train_samples, train_targets, test_samples, test_targets, folder_name):
    import os
    # import tikzplotlib
    import matplotlib.pyplot as plt

    from umap import UMAP
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from src.features.encodings import binary, nd, ncp, kmer, pse_knc
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    encoders = [
        binary.Encoder(),
        nd.Encoder(),
        ncp.Encoder(),
        kmer.Encoder(k=3),
        pse_knc.Encoder(),
    ]

    reducers = [
        (PCA(n_components=2), True),
        # LDA(n_components=2),
        (UMAP(n_components=2), False),
        (UMAP(n_components=2), True),
        (TSNE(n_components=2), False),
    ]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, encoder in enumerate(encoders):
        encoded_train_samples = encoder.fit_transform(train_samples, y=train_targets)
        encoded_test_samples = encoder.transform(test_samples)

        for j, item in enumerate(reducers):
            reducer, flag = item
            if flag:
                reduced_train_samples = reducer.fit_transform(encoded_train_samples, y=train_targets)
            else:
                reduced_train_samples = reducer.fit_transform(encoded_train_samples)

            plt.figure(figsize=(6, 6))
            plt.scatter(reduced_train_samples[:, 0], reduced_train_samples[:, 1], c=train_targets, cmap='viridis')
            plt.title(f'Encoder {i + 1}, Reducer {j + 1}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')

            plt.savefig(f"{folder_name}/plot_{i + 1}_{j + 1}.png")
            plt.close()


if __name__ == '__main__':
    from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch

    human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
    human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)

    mouse_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

    yeast_train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
    yeast_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)

    _generate_visualization_grid(human_train_dataset.samples, human_train_dataset.targets.values,
                                 human_test_dataset.samples,
                                 human_test_dataset.targets.values, 'human')
    # _generate_visualization_grid()
    # _generate_visualization_grid()
