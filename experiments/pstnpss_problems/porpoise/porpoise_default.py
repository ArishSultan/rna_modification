from src.utils import write_reports
from src.experiment.experiment import ExperimentFixed
from src.dataset import load_benchmark_dataset, Species, Modification

from src.model.porpoise.model import Factory
from src.model.porpoise.encoder import Encoder

human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi, False)

mouse_test_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, True)
mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)

yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

human_encoder = Encoder(Species.human)
mouse_encoder = Encoder(Species.mouse)
yeast_encoder = Encoder(Species.yeast)

human_experiment = ExperimentFixed(Factory(), human_test_dataset, human_train_dataset, human_encoder, k=5)
human_report = human_experiment.run()
write_reports(human_report, 'porpoise_default_human', Modification.psi.value, Species.human.value)

mouse_experiment = ExperimentFixed(Factory(), mouse_test_dataset, mouse_train_dataset, mouse_encoder, k=5)
mouse_report = mouse_experiment.run()
write_reports(mouse_report, 'porpoise_default_mouse', Modification.psi.value, Species.mouse.value)

yeast_experiment = ExperimentFixed(Factory(), yeast_test_dataset, yeast_train_dataset, yeast_encoder, k=5)
yeast_report = yeast_experiment.run()
write_reports(yeast_report, 'porpoise_default_yeast', Modification.psi.value, Species.yeast.value)
