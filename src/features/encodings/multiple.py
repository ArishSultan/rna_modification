from typing import Union
from pandas import DataFrame, concat
from sklearn.base import BaseEstimator, TransformerMixin


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoders: list[Union[BaseEstimator, TransformerMixin]]):
        self.encoders = encoders

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        encoded_frames = []
        for encoder in self.encoders:
            encoded_frames.append(encoder.fit_transform(x, **kwargs))

        return concat(encoded_frames, axis=1)

    def transform(self, x: DataFrame) -> DataFrame:
        encoded_frames = []
        for encoder in self.encoders:
            encoded_frames.append(encoder.transform(x))

        return concat(encoded_frames, axis=1)
