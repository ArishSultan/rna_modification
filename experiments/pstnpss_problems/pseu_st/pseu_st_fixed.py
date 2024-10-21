from src.model import pseu_st
from src.utils import write_reports
from src.experiment.experiment import ExperimentFixed
from sklearn.feature_selection import f_classif, SelectKBest
from src.dataset import load_benchmark_dataset, Species, Modification
from src.features.encodings import pstnpss, multiple, enac, ps, binary, ncp

encoder = multiple.Encoder([
    enac.Encoder(),
    pstnpss.Encoder(),
    binary.Encoder(),
    ncp.Encoder(),
    ps.Encoder(k=2),
    ps.Encoder(k=3)
])

human_feature_selector = SelectKBest(score_func=f_classif, k=129)
mouse_feature_selector = SelectKBest(score_func=f_classif, k=133)
yeast_feature_selector = SelectKBest(score_func=f_classif, k=306)

human_test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
human_train_dataset = load_benchmark_dataset(Species.human, Modification.psi, False)

mouse_test_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, True)
mouse_train_dataset = load_benchmark_dataset(Species.mouse, Modification.psi, False)

yeast_test_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, True)
yeast_train_dataset = load_benchmark_dataset(Species.yeast, Modification.psi, False)

human_experiment = ExperimentFixed(pseu_st.Factory(), human_test_dataset, human_train_dataset, encoder, k=10, feature_selector=human_feature_selector)
human_report = human_experiment.run()
write_reports(human_report, 'pstnpss_problem_pseu-st_fixed_human', Modification.psi.value, Species.human.value)

mouse_experiment = ExperimentFixed(pseu_st.Factory(), mouse_test_dataset, mouse_train_dataset, encoder, k=10, feature_selector=mouse_feature_selector)
mouse_report = mouse_experiment.run()
write_reports(mouse_report, 'pstnpss_problem_pseu-st_fixed_mouse', Modification.psi.value, Species.mouse.value)

yeast_experiment = ExperimentFixed(pseu_st.Factory(), yeast_test_dataset, yeast_train_dataset, encoder, k=10, feature_selector=yeast_feature_selector)
yeast_report = yeast_experiment.run()
write_reports(yeast_report, 'pstnpss_problem_pseu-st_fixed_yeast', Modification.psi.value, Species.yeast.value)
