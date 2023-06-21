from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(BaseEstimator, ClassifierMixin):
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass


class ModelFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_model(self) -> BaseModel:
        pass
