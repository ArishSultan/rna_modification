from pandas import read_csv, concat, DataFrame, Series

from src.model import lr
from src.utils import write_reports
from src.experiment import Experiment
from src.model.porpoise import pstnpss
from src.dataset import load_benchmark_dataset, Species, Modification

big_dataset = read_csv(
    '/Users/arish/Workspace/experiments/rna_modification/data.csv', header=None)

train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
big_sample_set = DataFrame(
    {'sequence': [
        *big_dataset[0].values,
        # *train_dataset.samples.values[:, 0], *test_dataset.samples.values[:, 0]
    ]})
big_target_set = [
    *big_dataset[1].values,
    # *train_dataset.targets.values, *test_dataset.targets.values
]

encoder = pstnpss.Encoder(Species.human)
encoded_samples_big = encoder.fit_transform(big_sample_set)

experiment = Experiment(lr.Factory(), test_dataset, train_dataset, encoder, k=10, should_fit_encoder=False)

report = experiment.run()

write_reports(report, 'reports/PSI PSTNPSS XGB', Modification.psi.value, Species.human.value)
