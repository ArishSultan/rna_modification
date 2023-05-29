from sklearn.utils import Bunch
from pandas import DataFrame, Series


class SeqBunch(Bunch):
    def __init__(self, samples: DataFrame, targets: Series, description: str, **kwargs):
        super().__init__(_samples=samples, _targets=targets, _description=description, **kwargs)

    @property
    def samples(self) -> DataFrame:
        return self._samples

    @property
    def targets(self) -> Series:
        return self._targets

    @property
    def description(self) -> str:
        return self._description
