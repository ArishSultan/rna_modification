from src.model import porpoise
from src.experiment import Experiment
from src.data import load_psi, Species
from src.experiment.reports import generate_latex_report, generate_kfold_latex_report

human_experiment = Experiment(
    porpoise.Factory(),
    load_psi(Species.human, independent=True),
    load_psi(Species.human, independent=False),
    porpoise.Encoder(Species.human),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'human_train', 'porpoise', True)
generate_latex_report(human_experiment['test'], 'human_test', 'porpoise', True)

yeast_experiment = Experiment(
    porpoise.Factory(),
    load_psi(Species.yeast, independent=True),
    load_psi(Species.yeast, independent=False),
    porpoise.Encoder(Species.yeast),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'yeast_train', 'porpoise', True)
generate_latex_report(human_experiment['test'], 'yeast_test', 'porpoise', True)

mouse_experiment = Experiment(
    porpoise.Factory(),
    None,
    load_psi(Species.mouse, independent=False),
    porpoise.Encoder(Species.mouse),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'mouse_train', 'porpoise', True)
