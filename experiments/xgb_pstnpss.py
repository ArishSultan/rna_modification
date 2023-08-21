from src.model import xgboost
from src.experiment import Experiment
from src.data import load_psi, Species
from src.features.encodings import pstnpss
from src.experiment.reports import generate_latex_report, generate_kfold_latex_report


human_experiment = Experiment(
    xgboost.Factory(),
    load_psi(Species.human, independent=True),
    load_psi(Species.human, independent=False),
    pstnpss.Encoder(),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'human_train', 'xgb_pstnpss', True)
generate_latex_report(human_experiment['test'], 'human_test', 'xgb_pstnpss', True)


yeast_experiment = Experiment(
    xgboost.Factory(),
    load_psi(Species.yeast, independent=True),
    load_psi(Species.yeast, independent=False),
    pstnpss.Encoder(),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'yeast_train', 'xgb_pstnpss', True)
generate_latex_report(human_experiment['test'], 'yeast_test', 'xgb_pstnpss', True)


mouse_experiment = Experiment(
    xgboost.Factory(),
    None,
    load_psi(Species.mouse, independent=False),
    pstnpss.Encoder(),
    k=10,
).run()

generate_kfold_latex_report(human_experiment['train'], 'mouse_train', 'xgb_pstnpss', True)

