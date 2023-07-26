from .base import BaseModel
from ..utils import get_list_indices
from ..features.encodings import pstnpss, binary, pse_knc
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from mlxtend.feature_selection import ColumnSelector
from mlxtend.classifier import StackingCVClassifier

_binary_selection = [46, 36, 44, 77, 45, 49, 3]
_pstnpss_selection = [10, 12, 8, 9, 17, 7, 6, 15, 14, 11, 2, 13, 5, 16, 18, 4, 1, 3, 0]
_pse_knc_selection = [4, 21, 20, 0, 5, 65, 64, 1, 17, 16, 60, 15, 25, 54, 3, 11, 46, 41, 40, 39, 7, 2, 42, 36,
                      31, 12, 24, 35, 18, 62, 61, 53, 8, 50, 56, 43, 19, 28, 9, 59, 37, 38, 33, 48, 44, 45, 63,
                      10, 22, 27, 49, 57, 55, 14, 51, 47, 52, 26, 23, 29, 30, 13, 6, 58]


class Porpoise(BaseModel):
    def __init__(self):
        selector0 = ColumnSelector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        classifier0 = XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bynode=1, max_depth=6, verbosity=1, colsample_bytree=0.637482,
            subsample=0.901284, learning_rate=0.276002, reg_alpha=0, max_delta_step=0, min_child_weight=1, n_jobs=1,
            n_estimators=1082, colsample_bylevel=1, random_state=0, reg_lambda=1, scale_pos_weight=1, gamma=0.103823
        )

        selector1 = ColumnSelector([19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
        classifier1 = GradientBoostingClassifier(
            learning_rate=0.247286, loss='log_loss', max_depth=9, n_estimators=1624, subsample=0.681402
        )

        selector2 = ColumnSelector([
            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
            92, 93, 94, 95, 96, 97, 98, 99, 100, 101
        ])
        classifier2 = GaussianNB()

        selector3 = ColumnSelector([102, 103, 104, 105, 106, 107, 108])
        classifier3 = SVC(C=26397.4411193282, gamma=0.0212845791037017, probability=True)

        self._model = StackingCVClassifier(
            cv=10,
            meta_classifier=LogisticRegression(multi_class='ovr', n_jobs=1, solver='liblinear'),
            classifiers=[
                Pipeline([('columns', selector0), ('xgb', classifier0)]),
                Pipeline([('columns', selector1), ('gradient_boost', classifier1)]),
                Pipeline([('columns', selector2), ('gaussian_naive_bayes', classifier2)]),
                Pipeline([('columns', selector3), ('svc', classifier3)]),
            ]
        )

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)

    def predict_proba(self, x):
        return self._model.predict_proba(x)

    @staticmethod
    def encode_seq(sequence: str, species: str, info=None):
        if info is None:
            info = pse_knc.get_info('PseKNC')

        binary_result = binary.encode(sequence)
        pstnpss_result = pstnpss.encode(sequence, species)
        pse_knc_result = pse_knc.encode(sequence, info, 3, 2, 0.1)

        return get_list_indices(pstnpss_result, _pstnpss_selection) + \
            get_list_indices(pstnpss_result, _pstnpss_selection) + \
            get_list_indices(pse_knc_result, _pse_knc_selection) + \
            get_list_indices(binary_result, _binary_selection)
