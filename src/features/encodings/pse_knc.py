from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from . import kmer
from ...utils.features import encode_df, get_info_file

_K_2_INDEX = {'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
              'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
              'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
              'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15}


def calculate_correlation(mer1, mer2, indexes, info):
    keys, values = info

    correlation = 0
    for key in keys:
        value = values[key]
        correlation += (float(value[indexes[mer1]]) - float(value[indexes[mer2]])) ** 2

    return correlation / len(keys)


def calculate_theta_list(info: tuple, indexes: dict, lambda0: float, sequence: str, k=2):
    theta_list = []

    for temp_lambda in range(int(lambda0)):
        theta = 0

        length = len(sequence) - temp_lambda - k
        for i in range(length):
            mer1 = sequence[i:i + k]
            mer2 = sequence[i + temp_lambda + 1: i + temp_lambda + 1 + k]

            theta += calculate_correlation(mer1, mer2, indexes, info)

        theta_list.append(theta / length)

    return theta_list


def get_info():
    return get_info_file('PseKNC')


def encode(sequence: str, info: tuple, k=2, lambda0: float = 1, weight: float = 0) -> list[float]:
    kmer_count = kmer.kmer_count(sequence, k, normalize=True)
    theta_list = calculate_theta_list(info, _K_2_INDEX, lambda0, sequence, 2)
    sum_theta_list = sum(theta_list)

    encoded_seq = []
    for mer in kmer.generate_all_kmers(k):
        encoded_seq.append(kmer_count[mer] / (1 + weight * sum_theta_list))

    base = 4 ** k + 1
    for k in range(base, base + lambda0):
        encoded_seq.append((weight * theta_list[k - base]) / (1 + weight * sum_theta_list))

    return encoded_seq


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, lambda0: float = 1, weight: float = 0):
        self.k = k
        self.weight = weight
        self.lambda0 = lambda0

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        info = get_info()
        return encode_df(x, lambda seq: encode(seq, info, self.k, self.lambda0, self.weight), 'pse_knc')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
