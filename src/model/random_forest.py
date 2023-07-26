from .base import ModelFactory
from sklearn.ensemble import RandomForestClassifier


class Factory(ModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    def create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier()
