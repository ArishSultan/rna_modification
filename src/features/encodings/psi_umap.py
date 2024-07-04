import umap

from typing import Union
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

# from ...utils.features import encode_df


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder: Union[BaseEstimator, TransformerMixin], n_components: int = 2):
        self._encoder = encoder
        self._n_components = n_components
        self._umap = umap.UMAP(n_components=n_components)

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        encoded_x = self._encoder.fit_transform(x, y=kwargs['y'])
        encoded_values = self._umap.fit_transform(encoded_x, y=kwargs['y'])

        return DataFrame(data=encoded_values, columns=[f'umap_{i}' for i in range(self._n_components)])

    def transform(self, x: DataFrame) -> DataFrame:
        encoded_x = self._encoder.transform(x)
        encoded_values = self._umap.transform(encoded_x)

        return DataFrame(data=encoded_values, columns=[f'umap_{i}' for i in range(self._n_components)])
