from .base import ModelFactory
from sklearn.linear_model import LogisticRegression


class Factory(ModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    def create_model(self) -> LogisticRegression:
        return LogisticRegression()
