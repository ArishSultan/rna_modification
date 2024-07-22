import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.dataset import Species
from src.experiment import ExperimentNew
from src.features.encodings import pse_knc as encoding

from src.utils.literature import literature_experiment

encoder = encoding.Encoder()
grid_search = GridSearchCV(
    cv=10,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy',
    estimator=SVC(),
    param_grid={
        'kernel': ['rbf'],
        'C': np.logspace(-5, 15, num=21, base=2),
        'gamma': np.logspace(-15, -5, num=11, base=2),
    },
)


def perform_op(train_dataset, test_dataset):
    grid_search.fit(encoder.fit_transform(train_dataset.samples), train_dataset.targets)

    experiment = ExperimentNew(lambda: SVC(probability=True, **grid_search.best_params_), encoder)
    fold_reports, mean_report = experiment.cross_validate(train_dataset.samples, train_dataset.targets)
    experiment.fit(train_dataset.samples, train_dataset.targets)

    if test_dataset is not None:
        test_report = experiment.evaluate(test_dataset.samples, test_dataset.targets)
    else:
        test_report = None

    return fold_reports, mean_report, test_report


literature_experiment('iRNA-PseU', Species.all(), perform_op)
