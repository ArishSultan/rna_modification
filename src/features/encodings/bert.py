import torch

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModel, BertConfig

from ...utils.features import encode_df


def encode(
        sequence: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        mode: str = 'max',
        replace: bool = False
) -> list[float]:
    if replace:
        sequence = sequence.replace('U', 'T')

    sequence = ' '.join(list(sequence))

    inputs = tokenizer(sequence, return_tensors='pt')['input_ids']
    hidden_states = model(inputs)[0]

    if mode == 'max':
        return torch.max(hidden_states[0], dim=0)[0].detach().numpy()
    elif mode == 'min':
        return torch.min(hidden_states[0], dim=0)[0].detach().numpy()

    return torch.mean(hidden_states[0], dim=0).detach().numpy()


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, model: str, mode: str = 'max', replace: bool = False):
        config = BertConfig.from_pretrained(model)

        self.mode = mode
        self.replace = replace
        self.model = AutoModel.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def fit_transform(self, x: DataFrame, **kwargs) -> DataFrame:
        return encode_df(x, lambda seq: encode(seq, self.tokenizer, self.model, self.mode, self.replace), 'bert')

    def transform(self, x: DataFrame) -> DataFrame:
        return self.fit_transform(x)
