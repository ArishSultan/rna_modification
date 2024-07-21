from umap import UMAP
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering, Birch

from src.dataset import load_benchmark_dataset, Species, Modification
from src.features.encodings import multiple, binary, ncp, pstnpss, pse_knc


def save_visualization(n_clusters, train_data, train_targets, train_clusters, test_data, test_targets, test_clusters,
                       visualization_dir: Path):
    visualization_dir.mkdir(parents=True, exist_ok=True)

    train_dir = visualization_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = visualization_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    reducer = UMAP(random_state=42, n_components=2, n_jobs=1)
    train_umap = reducer.fit_transform(train_data)
    test_umap = reducer.transform(test_data)

    for i in range(n_clusters):
        train_cluster = train_umap[train_clusters == i]
        train_cluster_targets = train_targets[train_clusters == i]

        test_cluster = test_umap[test_clusters == i]
        test_cluster_targets = test_targets[test_clusters == i]

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=train_cluster[:, 0], y=train_cluster[:, 1], hue=train_cluster_targets, palette='viridis')
        plt.title(f'UMAP - {n_clusters} Clusters')
        plt.savefig(train_dir / f'UMAP_{n_clusters}_{i}_train.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=test_cluster[:, 0], y=test_cluster[:, 1], hue=test_cluster_targets, palette='viridis')
        plt.title(f'UMAP - {n_clusters} Clusters')
        plt.savefig(test_dir / f'UMAP_{n_clusters}_{i}_test.png')
        plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=train_umap[:, 0], y=train_umap[:, 1], hue=train_targets, palette='viridis')
    plt.title(f'UMAP - {n_clusters} Clusters')
    plt.savefig(train_dir / f'UMAP_{n_clusters}_train.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=test_umap[:, 0], y=test_umap[:, 1], hue=test_targets, palette='viridis')
    plt.title(f'UMAP - {n_clusters} Clusters')
    plt.savefig(test_dir / f'UMAP_{n_clusters}_test.png')
    plt.close()


VISUALIZATION_DIR = Path('visualizations')
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

N_CLUSTERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ENCODERS = {
    'ncp': ncp.Encoder(),
    'binary': binary.Encoder(),
    'pstnpss': pstnpss.Encoder(),
    'pse_knc': pse_knc.Encoder(),

    'multiple': multiple.Encoder([
        binary.Encoder(),
        ncp.Encoder(),
        pstnpss.Encoder(),
        pse_knc.Encoder()
    ]),
}
CLUSTERS = {
    # 'kmeans': lambda n: KMeans(random_state=42, n_clusters=n),
    # 'gmm': lambda n: GaussianMixture(random_state=42, n_components=n),

    # 'dbscan': lambda n: DBSCAN(),
    # 'agglomerative': lambda n: AgglomerativeClustering(n_clusters=n),
    # 'meanshift': lambda n: MeanShift(bandwidth=n),
    # 'spectral': lambda n: SpectralClustering(random_state=42, n_clusters=n),
    'birch': lambda n: Birch(n_clusters=n)
}

for cluster in CLUSTERS:
    cluster_dir = VISUALIZATION_DIR / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for encoder in ENCODERS:
        encoder_dir = cluster_dir / encoder
        encoder_dir.mkdir(parents=True, exist_ok=True)

        _encoder = ENCODERS[encoder]
        train_samples = _encoder.fit_transform()
        test_samples = _encoder.transform(test_dataset.samples)

        for n_cluster in N_CLUSTERS:
            n_cluster_dir = encoder_dir / f'{n_cluster}_clusters'
            n_cluster_dir.mkdir(parents=True, exist_ok=True)

            cluster_algo_1 = CLUSTERS[cluster](n_cluster)
            train_clustered_1 = cluster_algo_1.fit_predict(train_samples)
            test_clustered_1 = cluster_algo_1.predict(test_samples)

            save_visualization(n_cluster, train_samples, train_dataset.targets, train_clustered_1, test_samples,
                               test_dataset.targets, test_clustered_1, n_cluster_dir)

            cluster_algo_2 = CLUSTERS[cluster](n_cluster)
            train_clustered_2 = cluster_algo_2.fit_predict(train_samples, train_dataset.targets)
            test_clustered_2 = cluster_algo_2.predict(test_samples)

            save_visualization(n_cluster, train_samples, train_dataset.targets, train_clustered_2, test_samples,
                               test_dataset.targets, test_clustered_2, n_cluster_dir / 'target')

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.mixture import GaussianMixture

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.manifold import TSNE
# import umap
#
# from src.dataset import load_benchmark_dataset, Species, Modification
# from src.features.encodings import multiple, binary, ncp, pstnpss, pse_knc
#
# train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
# test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
#
# encoder = multiple.Encoder(encoders=[
#     binary.Encoder(),
#     ncp.Encoder(),
#     pstnpss.Encoder(),
#     pse_knc.Encoder()
# ])
#
# train_y = train_dataset.targets
# train_x = encoder.fit_transform(train_dataset.samples, y=train_y)
# #
# test_y = test_dataset.targets
# test_x = encoder.transform(test_dataset.samples)
#
#
# # Function to create directories if they don't exist
# def create_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#
# # Function to plot and save UMAP and t-SNE visualizations
# def plot_clusters(data, labels, title, output_dir, cluster_name, n_clusters):
#     create_dir(output_dir)
#
#     # UMAP
#     reducer = umap.UMAP(random_state=42)
#     embedding = reducer.fit_transform(data)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='viridis')
#     plt.title(f'UMAP - {title} - {n_clusters} Clusters')
#     plt.savefig(os.path.join(output_dir, f'UMAP_{cluster_name}_{n_clusters}.png'))
#     plt.close()
#
#     # t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_result = tsne.fit_transform(data)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='viridis')
#     plt.title(f't-SNE - {title} - {n_clusters} Clusters')
#     plt.savefig(os.path.join(output_dir, f'tSNE_{cluster_name}_{n_clusters}.png'))
#     plt.close()
#
#
# # Function to apply clustering, classify within clusters, and visualize results
# def cluster_classify_visualize(train_x, train_y, test_x, test_y, clustering_algorithms, n_clusters_range):
#     for cluster_name, cluster_model in clustering_algorithms.items():
#         for n_clusters in n_clusters_range:
#             if cluster_name in ['dbscan', 'meanshift']:
#                 clusters_train = cluster_model.fit_predict(train_x)
#                 clusters_test = cluster_model.fit_predict(test_x)
#             else:
#                 cluster_model.set_params(n_clusters=n_clusters)
#                 clusters_train = cluster_model.fit_predict(train_x)
#                 clusters_test = cluster_model.predict(test_x)
#
#             output_dir = os.path.join("visualizations", cluster_name, f"{n_clusters}_clusters")
#             create_dir(output_dir)
#
#             plot_clusters(train_x, train_y, 'Train Data', output_dir, cluster_name, n_clusters)
#             plot_clusters(test_x, test_y, 'Test Data', output_dir, cluster_name, n_clusters)
#
#             cluster_classifiers = []
#             for cluster_id in range(n_clusters):
#                 cluster_train_x = train_x[clusters_train == cluster_id]
#                 cluster_train_y = train_y[clusters_train == cluster_id]
#
#                 if len(cluster_train_y) == 0:
#                     print(f"No data for cluster {cluster_id}. Skipping.")
#                     cluster_classifiers.append(None)
#                     continue
#
#                 clf = RandomForestClassifier().fit(cluster_train_x, cluster_train_y)
#                 cluster_classifiers.append(clf)
#
#             for cluster_id in range(n_clusters):
#                 cluster_test_x = test_x[clusters_test == cluster_id]
#                 cluster_test_y = test_y[clusters_test == cluster_id]
#
#                 if cluster_classifiers[cluster_id] is None or len(cluster_test_y) == 0:
#                     print(f"No data for cluster {cluster_id}. Skipping.")
#                     continue
#
#                 predictions = cluster_classifiers[cluster_id].predict(cluster_test_x)
#                 print(
#                     f"Classification report for cluster {cluster_id} with {n_clusters} clusters using {cluster_name}:")
#                 print(classification_report(cluster_test_y, predictions))
#
#
# # Example usage:
# # Assuming train_x, train_y, test_x, and test_y are already defined and loaded
#
# # Define the clustering algorithms
# clustering_algorithms = {
#     'kmeans': KMeans(random_state=42),
#     'gmm': GaussianMixture(random_state=42),
#     'dbscan': DBSCAN(),
#     'agglomerative': AgglomerativeClustering(),
#     'meanshift': MeanShift(),
#     'spectral': SpectralClustering(random_state=42),
#     'birch': Birch()
# }
#
# # Define the range of number of clusters
# n_clusters_range = range(3, 11)
#
# # Call the function
# cluster_classify_visualize(train_x, train_y, test_x, test_y, clustering_algorithms, n_clusters_range)
