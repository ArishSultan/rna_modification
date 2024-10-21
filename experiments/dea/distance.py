import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from src.features.encodings import binary, nd, ncp, kmer, pse_knc


def calculate_group_distances(train_samples, train_targets, test_samples=None, test_targets=None):
    # Split train samples into positive and negative classes
    train_pos = train_samples[train_targets == 1]
    train_neg = train_samples[train_targets == 0]

    test_pos = None
    test_neg = None
    if test_samples is not None and test_targets is not None:
        # Split test samples into positive and negative classes
        test_pos = test_samples[test_targets == 1]
        test_neg = test_samples[test_targets == 0]

    # Apply scaling for robustness
    scaler = StandardScaler()
    all_samples = np.vstack([train_samples, test_samples]) if test_samples is not None else train_samples
    all_samples_scaled = scaler.fit_transform(all_samples)

    # Re-split scaled data
    train_pos_scaled = all_samples_scaled[:len(train_pos)] if len(train_pos) > 0 else None
    train_neg_scaled = all_samples_scaled[len(train_pos):len(train_pos) + len(train_neg)] if len(
        train_neg) > 0 else None

    if test_samples is not None:
        test_pos_scaled = all_samples_scaled[
                          len(train_pos) + len(train_neg):len(train_pos) + len(train_neg) + len(test_pos)] if len(
            test_pos) > 0 else None
        test_neg_scaled = all_samples_scaled[len(train_pos) + len(train_neg) + len(test_pos):] if len(
            test_neg) > 0 else None
    else:
        test_pos_scaled = None
        test_neg_scaled = None

    # Calculate distances between different groups
    distances = {}
    if train_pos_scaled is not None and train_neg_scaled is not None:
        distances['train_pos_train_neg'] = np.mean(
            [euclidean(p, n) for p in train_pos_scaled for n in train_neg_scaled])

    if test_pos_scaled is not None and test_neg_scaled is not None:
        distances['train_pos_test_pos'] = np.mean([euclidean(p, t) for p in train_pos_scaled for t in test_pos_scaled])
        distances['train_pos_test_neg'] = np.mean([euclidean(p, t) for p in train_pos_scaled for t in test_neg_scaled])
        distances['train_neg_test_pos'] = np.mean([euclidean(n, t) for n in train_neg_scaled for t in test_pos_scaled])
        distances['train_neg_test_neg'] = np.mean([euclidean(n, t) for n in train_neg_scaled for t in test_neg_scaled])
        distances['test_pos_test_neg'] = np.mean(
            [euclidean(t1, t2) for t1 in test_pos_scaled for t2 in test_neg_scaled])

    return distances


def write_latex_table(species, encoding_name, distances, folder_name='latex_tables'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    table_filename = f"{folder_name}/{species}_{encoding_name}.tex"
    with open(table_filename, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Distance Matrix for {species.capitalize()} using {encoding_name} encoding}}\n")
        f.write("\\begin{tabular}{|l|l|}\n")
        f.write("\\hline\n")
        f.write("Distance Type & Value \\\\ \\hline\n")

        for key, value in distances.items():
            f.write(f"{key.replace('_', ' ')} & {value:.4f} \\\\ \\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def generate_distance_matrices(train_samples, train_targets, test_samples=None, test_targets=None, species='unknown',
                               folder_name='latex_tables'):
    encoders = [
        ('binary', binary.Encoder()),
        ('nd', nd.Encoder()),
        ('ncp', ncp.Encoder()),
        ('kmer', kmer.Encoder(k=3)),
        ('pse_knc', pse_knc.Encoder()),
    ]

    for encoding_name, encoder in encoders:
        # Encode the data
        encoded_train_samples = encoder.fit_transform(train_samples, y=train_targets)
        encoded_test_samples = encoder.transform(test_samples) if test_samples is not None else None

        # Calculate distances
        distances = calculate_group_distances(encoded_train_samples, train_targets, encoded_test_samples, test_targets)

        # Write distances to LaTeX table
        write_latex_table(species, encoding_name, distances, folder_name)


if __name__ == '__main__':
    from src.dataset import load_benchmark_dataset, Species, Modification

    # Load datasets for different species
    species_list = [Species.human, Species.mouse, Species.yeast]

    for species in species_list:
        train_dataset = load_benchmark_dataset(species, Modification.psi)

        # Check if test dataset exists
        try:
            test_dataset = load_benchmark_dataset(species, Modification.psi, True)
        except FileNotFoundError:
            test_dataset = None

        # Generate distance matrices for each species
        if test_dataset:
            generate_distance_matrices(train_dataset.samples, train_dataset.targets.values,
                                       test_dataset.samples, test_dataset.targets.values, species=species.name)
        else:
            # Only use train dataset
            generate_distance_matrices(train_dataset.samples, train_dataset.targets.values, species=species.name)