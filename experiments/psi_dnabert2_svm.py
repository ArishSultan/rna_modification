# from pandas import read_csv
#
# from src.model import xgboost
# from src.utils import write_reports
# from src.experiment import Experiment
# import pandas as pd
# from src.features.encodings import multiple, ncp, pstnpss, pse_knc, binary, bert
# from src.dataset import load_benchmark_dataset, Species, Modification, SeqBunch
#
# # dataset = load_benchmark_dataset(Species.human, Modification.psi)
# # dataset_test = load_benchmark_dataset(Species.human, Modification.psi, True)
# #
# # encoder = multiple.Encoder(encoders=[
# #     binary.Encoder(),
# #     ncp.Encoder(),
# #     pstnpss.Encoder(),
# #     pse_knc.Encoder()
# # ])
#
# # encoded_samples = encoder.fit_transform(dataset.samples, y=dataset.targets)
# # pd.concat([dataset.targets, encoded_samples], axis=1).to_csv('xx.csv', index=False)
# # experiment = Experiment(xgboost.Factory(), dataset_test, dataset, encoder, k=10)
# # #
# # #
# encoder = bert.Encoder("zhihan1996/DNABERT-2-117M", replace=True)
# human_dataset = load_benchmark_dataset(Species.human, Modification.psi)
# human_dataset_test = load_benchmark_dataset(Species.human, Modification.psi, True)
#
# experiment = Experiment(xgboost.Factory(), human_dataset_test, human_dataset, encoder, k=10)
# report = experiment.run()
# # #
# write_reports(report, 'PSI PSTNPSS Ensemble', Modification.psi.value, Species.human.value)
# # # ACGTGTACGTGCACGGACGACTAGTCAGCA
# #
# # # print(len('ACGTGTACGTGCACGGACGACTAGTCAGCA'))
