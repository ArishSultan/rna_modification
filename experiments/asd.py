from src.model import xgboost
from src.experiment import Experiment
from src.features.encodings import pstnpss_bi
from src.dataset import load_benchmark_dataset, Species, Modification
from src.utils import write_reports

dataset = load_benchmark_dataset(Species.human, Modification.psi)
dataset_test = load_benchmark_dataset(Species.human, Modification.psi, True)

encoder = pstnpss_bi.Encoder(3, 0)
encoded_samples = encoder.fit_transform(dataset.samples, y=dataset.targets)

experiment = Experiment(xgboost.Factory(), dataset_test, dataset, encoder, k=10)
report = experiment.run()

write_reports(report, 'PSI Human BiPSTP SVM', Modification.psi.value, Species.human.value)
#
#
# m6a_encoder = pstnpss.Encoder()
# human_dataset = load_dataset(Species.human, Modification.m6a)
#
# experiment = Experiment(xgboost.Factory(), None, human_dataset, m6a_encoder, k=10)
# report = experiment.run()
#
# write_reports(report, 'M6A PSTNPSS Ensemble', Modification.m6a.value, Species.human.value)
