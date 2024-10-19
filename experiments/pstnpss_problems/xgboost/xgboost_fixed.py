from src.experiment.experiment import ExperimentFixed
from src.model import xgboost
from src.utils import write_reports
from src.experiment import Experiment
from src.features.encodings import pstnpss
from src.dataset import load_benchmark_dataset, Species, Modification

human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi, False)

mouse_test_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, True)
mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)

yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

encoder = pstnpss.Encoder()

human_experiment = ExperimentFixed(xgboost.Factory(), human_test_dataset, human_train_dataset, encoder, k=10)
human_report = human_experiment.run()
write_reports(human_report, 'pstnpss_problem_xgb_human', Modification.psi.value, Species.human.value)

mouse_experiment = ExperimentFixed(xgboost.Factory(), mouse_test_dataset, mouse_train_dataset, encoder, k=10)
mouse_report = mouse_experiment.run()
write_reports(mouse_report, 'pstnpss_problem_xgb_mouse', Modification.psi.value, Species.mouse.value)

yeast_experiment = ExperimentFixed(xgboost.Factory(), yeast_test_dataset, yeast_train_dataset, encoder, k=10)
yeast_report = yeast_experiment.run()
write_reports(yeast_report, 'pstnpss_problem_xgb_yeast', Modification.psi.value, Species.yeast.value)
