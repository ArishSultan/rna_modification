from pandas import read_csv

from src.utils import write_reports
from src.experiment.experiment import ExperimentFixed
from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch

from src.model.svm import Factory
from src.model.porpoise.encoder import EncoderFixed

human_train_data = read_csv('/Users/arish/Research/research/rna_modification/dataset/train_set.csv', header=None)
human_test_data = read_csv('/Users/arish/Research/research/rna_modification/dataset/test_set.csv', header=None)

human_test_dataset = SeqBunch(
    targets=human_test_data[1],
    samples=human_test_data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
)
# load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = SeqBunch(
    targets=human_train_data[1],
    samples=human_train_data.drop(1, axis=1).rename({0: 'sequence'}, axis=1),
)

# load_benchmark_dataset(Species.human, Modification.psi, False))

# mouse_test_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, True)
# mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)
#
# yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
# yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

encoder = EncoderFixed()

human_experiment = ExperimentFixed(Factory(), human_test_dataset, human_train_dataset, encoder, k=5)
human_report = human_experiment.run()
write_reports(human_report, 'svm_fixed_external_human', Modification.psi.value, Species.human.value)
#
# mouse_experiment = ExperimentFixed(Factory(), mouse_test_dataset, mouse_train_dataset, encoder, k=5)
# mouse_report = mouse_experiment.run()
# write_reports(mouse_report, 'porpoise_fixed_mouse', Modification.psi.value, Species.mouse.value)

# yeast_experiment = ExperimentFixed(Factory(), yeast_test_dataset, yeast_train_dataset, encoder, k=5)
# yeast_report = yeast_experiment.run()
# write_reports(yeast_report, 'porpoise_fixed_yeast', Modification.psi.value, Species.yeast.value)
