from pathlib import Path
from collections import Counter
from src.dataset import load_benchmark_dataset, Species, Modification

import matplotlib.pyplot as plt

from src.utils import get_dump_path


def nucleotide_composition_viz(sequences: list[str], labels: list[int], k: int, out_name: Path):
    # Separate positive and negative sequences based on labels
    positive_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
    negative_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]

    # Count k-mers in both positive and negative sequences
    def count_kmers(sequences, k):
        kmers = []
        for seq in sequences:
            kmers.extend([seq[i:i + k] for i in range(len(seq) - k + 1)])
        return Counter(kmers)

    pos_kmer_counts = count_kmers(positive_sequences, k)
    neg_kmer_counts = count_kmers(negative_sequences, k)

    # Get the union of k-mers from both positive and negative sets
    all_kmers = set(pos_kmer_counts.keys()).union(set(neg_kmer_counts.keys()))

    # Sort k-mers for consistent plotting
    all_kmers = sorted(all_kmers)

    # Create bar plots for positive and negative counts
    pos_counts = [pos_kmer_counts[kmer] for kmer in all_kmers]
    neg_counts = [neg_kmer_counts[kmer] for kmer in all_kmers]

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    indices = range(len(all_kmers))

    # Plot positive bars
    plt.bar(indices, pos_counts, bar_width, label='Positive', alpha=0.7)

    # Plot negative bars with some offset
    plt.bar([i + bar_width for i in indices], neg_counts, bar_width, label='Negative', alpha=0.7)

    # Labels and title
    plt.xticks([i + bar_width / 2 for i in indices], all_kmers, rotation=90)
    plt.grid(True)
    plt.legend()

    out_name.parent.mkdir(exist_ok=True, parents=True)

    # Save the plot
    plt.tight_layout()
    import tikzplotlib
    tikzplotlib.save(out_name)
    plt.close()


human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi, False)

mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)

yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

# nc_path = get_dump_path(Path('dea') / 'nc')
nc_path = Path('/Users/arish/Research/research/rna_modification/writing/chapter/images/tikz')
nucleotide_composition_viz(human_train_dataset.samples['sequence'].values, human_train_dataset.targets.values, k=1,
                           out_name=nc_path / 'nc_h_990(k=1).tex')
nucleotide_composition_viz(human_train_dataset.samples['sequence'].values, human_train_dataset.targets.values, k=2,
                           out_name=nc_path / 'nc_h_990(k=2).tex')
nucleotide_composition_viz(human_train_dataset.samples['sequence'].values, human_train_dataset.targets.values, k=3,
                           out_name=nc_path / 'nc_h_990(k=3).tex')
nucleotide_composition_viz(human_test_dataset.samples['sequence'].values, human_test_dataset.targets.values, k=1,
                           out_name=nc_path / 'nc_h_200(k=1).tex')
nucleotide_composition_viz(human_test_dataset.samples['sequence'].values, human_test_dataset.targets.values, k=2,
                           out_name=nc_path / 'nc_h_200(k=2).tex')
nucleotide_composition_viz(human_test_dataset.samples['sequence'].values, human_test_dataset.targets.values, k=3,
                           out_name=nc_path / 'nc_h_200(k=3).tex')

nucleotide_composition_viz(mouse_train_dataset.samples['sequence'].values, mouse_train_dataset.targets.values, k=1,
                           out_name=nc_path / 'nc_m_944(k=1).tex')
nucleotide_composition_viz(mouse_train_dataset.samples['sequence'].values, mouse_train_dataset.targets.values, k=2,
                           out_name=nc_path / 'nc_m_944(k=2).tex')
nucleotide_composition_viz(mouse_train_dataset.samples['sequence'].values, mouse_train_dataset.targets.values, k=3,
                           out_name=nc_path / 'nc_m_944(k=3).tex')

nucleotide_composition_viz(yeast_train_dataset.samples['sequence'].values, yeast_train_dataset.targets.values, k=1,
                           out_name=nc_path / 'nc_s_628(k=1).tex')
nucleotide_composition_viz(yeast_train_dataset.samples['sequence'].values, yeast_train_dataset.targets.values, k=2,
                           out_name=nc_path / 'nc_s_628(k=2).tex')
nucleotide_composition_viz(yeast_train_dataset.samples['sequence'].values, yeast_train_dataset.targets.values, k=3,
                           out_name=nc_path / 'nc_s_628(k=3).tex')
nucleotide_composition_viz(yeast_test_dataset.samples['sequence'].values, yeast_test_dataset.targets.values, k=1,
                           out_name=nc_path / 'nc_s_200(k=1).tex')
nucleotide_composition_viz(yeast_test_dataset.samples['sequence'].values, yeast_test_dataset.targets.values, k=2,
                           out_name=nc_path / 'nc_s_200(k=2).tex')
nucleotide_composition_viz(yeast_test_dataset.samples['sequence'].values, yeast_test_dataset.targets.values, k=3,
                           out_name=nc_path / 'nc_s_200(k=3).tex')