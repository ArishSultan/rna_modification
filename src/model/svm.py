from .base import ModelFactory
from sklearn.svm import SVC


class Factory(ModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    def create_model(self) -> SVC:
        return SVC(probability=True)
