from src.model import svm as model, BasicFactoryModel
from src.experiment import Experiment
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from src.features.encodings import bi_pstp as enc
from src.dataset import load_benchmark_dataset, Species, Modification
from src.utils import write_reports

dataset = load_benchmark_dataset(Species.human, Modification.psi)
dataset_test = load_benchmark_dataset(Species.human, Modification.psi, True)

encoder = enc.Encoder(k=3)
encoded_samples = encoder.fit_transform(dataset.samples, y=dataset.targets)
encoded_samples_test = encoder.transform(dataset_test.samples)

param_grid = {
    'C': [2**i for i in range(-5, 16)],
    'gamma': [2**i for i in range(-15, 6)]
}

# Initialize the SVM model with RBF kernel
svm_model = svm.SVC(kernel='rbf')

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(encoded_samples, dataset.targets)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_index_)

print(grid_search.best_estimator_.score(encoded_samples_test, dataset_test.targets))

# best_model = grid_search.best_estimator_
#
# experiment = Experiment(BasicFactoryModel(best_model), dataset_test, dataset, encoder, k=10)
# report = experiment.run()
#
# write_reports(report, 'PSI Human BiPSTP fixed SVM', Modification.psi.value, Species.human.value)
#
#
# m6a_encoder = pstnpss.Encoder()
# human_dataset = load_dataset(Species.human, Modification.m6a)
#
# experiment = Experiment(xgboost.Factory(), None, human_dataset, m6a_encoder, k=10)
# report = experiment.run()
#
# write_reports(report, 'M6A PSTNPSS Ensemble', Modification.m6a.value, Species.human.value)
