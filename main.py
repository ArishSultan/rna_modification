from pandas import DataFrame

from src.dataset import load_dataset, Species, Modification

dataset = load_dataset(Species.human, Modification.psi)

targets = dataset.targets.values
sequences = dataset.samples.sequence.values

new_sequences = []
new_targets = []
for i in range(len(sequences)):
    new_targets.append(targets[i])
    new_sequences.append(sequences[i][10:-10])

DataFrame({'sequence': new_sequences, 'targets': new_targets}).to_csv('data.csv', header=False, index=False)