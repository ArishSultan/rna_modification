from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import ColumnSelector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from ..base import ModelFactory, BaseModel


class HumanAndMouseModel(BaseModel):
    def __init__(self):
        self._model = StackingClassifier(
            meta_classifier=LogisticRegression(),
            classifiers=[RandomForestClassifier(n_estimators=100,random_state=100), LogisticRegression()],
        )

    def fit(self, x, y):
        return self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)

    def predict_proba(self, x):
        return self._model.predict_proba(x)


class YeastModel(BaseModel):
    def __init__(self):
        self._model = StackingClassifier(
            meta_classifier=LogisticRegression(),
            classifiers=[RandomForestClassifier(n_estimators=100,random_state=100), LogisticRegression()],
        )

    def fit(self, x, y):
        return self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)

    def predict_proba(self, x):
        return self._model.predict_proba(x)


class Factory(ModelFactory):
    def create_model(self) -> BaseModel:
        return HumanAndMouseModel()
