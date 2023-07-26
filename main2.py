from pathlib import Path

from src.model import random_forest
from src.data import load_psi, Species
from src.features.encodings import pstnpss
from src.experiment.experiment import Report, Experiment
from src.experiment.reports.latex_report import *

from sklearn.ensemble import RandomForestClassifier

train_ds = load_psi(Species.yeast, False)
test_ds = load_psi(Species.yeast, True)

encoder = pstnpss.Encoder(Species.yeast)

report = Experiment(random_forest.Factory(), test_ds, train_ds, encoder).run()

# classifier = RandomForestClassifier()
# classifier.fit(x, y)
#
generate_kfold_latex_report(report['train'], name='train', out=Path('/Users/arish/Workspace/research/rna_modification/notebooks/experiments/sklearn/pseknc_yeast'), generate_pdf=True)
generate_latex_report(report['test'], name='test', out=Path('/Users/arish/Workspace/research/rna_modification/notebooks/experiments/sklearn/pseknc_yeast'), generate_pdf=True)
#
#
#

