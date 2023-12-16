from sklearn.utils import Bunch
from pandas import DataFrame, Series
from sklearn.model_selection import KFold


class SeqFoldBunch(Bunch):
    def __init__(self, samples: DataFrame, targets: Series, **kwargs):
        super().__init__(_samples=samples, _targets=targets, **kwargs)

    @property
    def samples(self) -> DataFrame:
        return self._samples

    @property
    def targets(self) -> Series:
        return self._targets


class SeqBunch(Bunch):
    def __init__(self, samples: DataFrame, targets: Series, **kwargs):
        super().__init__(_samples=samples, _targets=targets, **kwargs)

    @property
    def samples(self) -> DataFrame:
        return self._samples

    @property
    def targets(self) -> Series:
        return self._targets

    def k_fold(self, k=5):
        kf = KFold(n_splits=k)

        for train_index, test_index in kf.split(self.samples):
            yield SeqFoldBunch(
                samples=self.samples.iloc[train_index],
                targets=self.targets.iloc[train_index],
            ), SeqFoldBunch(
                samples=self.samples.iloc[test_index],
                targets=self.targets.iloc[test_index],
            )
