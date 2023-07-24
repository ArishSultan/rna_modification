from pathlib import Path
from src.data import load_psi, Species
from src.features import anf
from src.model import lr
from src.experiment import Experiment
from src.experiment.reports.latex_report import generate_latex_report

test_dataset = load_psi(Species.human, independent=True)
train_dataset = load_psi(Species.human, independent=False)

encoder = anf.Encoder()

factory = lr.Factory()

report = Experiment(factory, test_dataset, train_dataset, encoder).run()

generate_latex_report(report['test'], name=Path('asd'), assets_dir=None)
