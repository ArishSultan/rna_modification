from .base import ModelFactory
from xgboost import XGBClassifier


class Factory(ModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = kwargs

    def create_model(self) -> XGBClassifier:
        return XGBClassifier()
