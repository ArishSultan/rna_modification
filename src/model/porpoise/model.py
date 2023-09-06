from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import ColumnSelector
from sklearn.ensemble import GradientBoostingClassifier

from ..base import ModelFactory, BaseModel


class Model(BaseModel):
    def __init__(self):
        # s0 = ColumnSelector(range(19))
        # c0 = XGBClassifier(
        #     base_score=0.5, booster='gbtree', colsample_bynode=1, max_depth=6, verbosity=1, colsample_bytree=0.637482,
        #     subsample=0.901284, learning_rate=0.276002, reg_alpha=0, max_delta_step=0, min_child_weight=1, n_jobs=1,
        #     n_estimators=1082, colsample_bylevel=1, random_state=0, reg_lambda=1, scale_pos_weight=1, gamma=0.103823
        # )

        s1 = ColumnSelector(range(19 - 19, 38 - 19))
        c1 = GradientBoostingClassifier(
            learning_rate=0.247286, loss='log_loss', max_depth=9,
            n_estimators=1624, subsample=0.681402,
        )

        s2 = ColumnSelector(range(38 - 19, 102 - 19))
        c2 = GaussianNB()

        s3 = ColumnSelector(range(102 - 19, 109 - 19))
        c3 = SVC(C=26397.4411193282, gamma=0.0212845791037017, probability=True)

        self._model = StackingClassifier(
            meta_classifier=LogisticRegression(multi_class='ovr', n_jobs=1, solver='liblinear'),
            classifiers=[
                # Pipeline([('columns', s0), ('xgb', c0)]),
                Pipeline([('columns', s1), ('gradient_boost', c1)]),
                Pipeline([('columns', s2), ('gaussian_naive_bayes', c2)]),
                Pipeline([('columns', s3), ('svc', c3)]),
            ]
        )

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)

    def predict_proba(self, x):
        return self._model.predict_proba(x)


class Factory(ModelFactory):
    def create_model(self) -> BaseModel:
        return Model()
