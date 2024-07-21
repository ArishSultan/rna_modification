from abc import ABC, abstractmethod
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin


class BaseEncoder(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, x: DataFrame, y: Series):
        """
        """
        pass

    @abstractmethod
    def transform(self, x: DataFrame):
        """
        """
        pass

    @abstractmethod
    def fit_transform(self, x: DataFrame, **kwargs):
        """
        """
        pass
