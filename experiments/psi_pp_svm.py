# # from pandas import read_csv
import RNA
# #
from src.model import xgboost, svm
from src.utils import write_reports
from src.experiment import Experiment
from src.features.encodings import bipstp
from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch, load_dataset

dataset = load_benchmark_dataset(Species.human, Modification.psi, False)
test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
# print(prob.Encoder().fit_transform(dataset.samples, y=dataset.targets).iloc[0])
# for item in range(len(dataset.targets)):
#     if dataset.targets[item] == 1:
#         print(RNA.fold(dataset.samples.values[item][0]))
#
experiment = Experiment(svm.Factory(), test_dataset, dataset, bipstp.Encoder(), k=10)

report = experiment.run()

write_reports(report, 'PSI Bipstp SVM', Modification.psi.value, Species.human.value)
#
# fold1 = RNA.fold_compound('CAUGGAGAGAUGUUCUUUACU')
# fold2 = RNA.fold_compound('CAAGUCGGCUUUGCUAUAAAC')
#
# print(RNA.fold('CAUGGAGAGAUGUUCUUUACU'))
# print(RNA.fold('CAUAUCAACUUUUAUUCUCUC'))
# print(fold2.mfe())
# # print(ViennaRNA.Make_swString('CAUGGAGAGAUGUUCUUUACU'))
# #
# # print(ViennaRNA.rna.string_edit_distance(
# #     ViennaRNA.Make_swString('CAUGGAGAGAUGUUCUUUACU'),
# #     ViennaRNA.Make_swString('CAAGUCGGCUUUGCUAUAAAC')
# # ))