from umap import UMAP
from src.model import svm, xgboost, lr, random_forest
from src.experiment import Experiment
from src.features.encodings import multiple, binary, ncp, pstnpss, pse_knc, psi_umap
from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch
from src.utils import write_reports

# Encoder definition
encoder = multiple.Encoder([
    binary.Encoder(),
    ncp.Encoder(),
    pstnpss.Encoder(),
    pse_knc.Encoder()
])

# UMAP transformer
umap_transformer = UMAP(n_components=100, random_state=42)

# Load datasets
test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
train_dataset = load_benchmark_dataset(Species.human, Modification.psi)
#
# # Transform datasets
# train_samples = encoder.fit_transform(train_dataset.samples, y=train_dataset.targets)
# train_samples = umap_transformer.fit_transform(train_samples, y=train_dataset.targets)
#
# test_samples = encoder.transform(test_dataset.samples)
# test_samples = umap_transformer.transform(test_samples)


experiment = Experiment(svm.Factory(), test_dataset, train_dataset, psi_umap.Encoder(psi_umap.Encoder(encoder), n_components=3), k=10)
report = experiment.run()
write_reports(report, 'gmm_umap/SVM', Modification.psi.value, Species.human.value)

experiment = Experiment(xgboost.Factory(), test_dataset, train_dataset, psi_umap.Encoder(psi_umap.Encoder(encoder), n_components=3), k=10)
report = experiment.run()
write_reports(report, 'gmm_umap/XGBoost', Modification.psi.value, Species.human.value)

experiment = Experiment(lr.Factory(), test_dataset, train_dataset, psi_umap.Encoder(psi_umap.Encoder(encoder), n_components=3), k=10)
report = experiment.run()
write_reports(report, 'gmm_umap/LR', Modification.psi.value, Species.human.value)

experiment = Experiment(random_forest.Factory(), test_dataset, train_dataset, psi_umap.Encoder(psi_umap.Encoder(encoder), n_components=3), k=10)
report = experiment.run()
write_reports(report, 'gmm_umap/RF', Modification.psi.value, Species.human.value)
