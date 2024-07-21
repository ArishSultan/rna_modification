from pandas import DataFrame, concat, Series

from ..encoder import BaseEncoder


class Encoder(BaseEncoder):
    def __init__(self, encoders: list[BaseEncoder]):
        self.encoders = encoders

    def fit(self, x: DataFrame, y: Series):
        for encoder in self.encoders:
            encoder.fit(x, y)

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        self.fit(x, kwargs['y'])
        return self.fit_transform(x)

    def transform(self, x: DataFrame) -> DataFrame:
        encoded_frames = []
        for encoder in self.encoders:
            encoded_frames.append(encoder.transform(x))

        return concat(encoded_frames, axis=1)
