from src.utils import write_reports
from src.experiment.experiment import ExperimentFixed
from src.dataset import load_benchmark_dataset, Species, Modification

from src.model.porpoise.model import Factory
from src.model.porpoise.encoder import EncoderFixed

human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi, False)

mouse_test_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, True)
mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)

yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

encoder = EncoderFixed()

human_experiment = ExperimentFixed(Factory(), human_test_dataset, human_train_dataset, encoder, k=5)
human_report = human_experiment.run()
write_reports(human_report, 'porpoise_fixed_human', Modification.psi.value, Species.human.value)

mouse_experiment = ExperimentFixed(Factory(), mouse_test_dataset, mouse_train_dataset, encoder, k=5)
mouse_report = mouse_experiment.run()
write_reports(mouse_report, 'porpoise_fixed_mouse', Modification.psi.value, Species.mouse.value)

yeast_experiment = ExperimentFixed(Factory(), yeast_test_dataset, yeast_train_dataset, encoder, k=5)
yeast_report = yeast_experiment.run()
write_reports(yeast_report, 'porpoise_fixed_yeast', Modification.psi.value, Species.yeast.value)
