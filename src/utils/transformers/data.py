from ...dataset import SeqBunch

from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer, BatchEncoding


def encode_sequence(sequence: str, tokenizer: PreTrainedTokenizer, split_chars=True):
    max_length = len(sequence)

    if split_chars:
        sequence = ' '.join(list(sequence))

    return tokenizer(sequence, max_length=max_length, add_special_tokens=False)


def encode_seq_bunch(
        bunch: SeqBunch,
        tokenizer: PreTrainedTokenizer,
        split_chars=True
) -> tuple[list[BatchEncoding], list[int]]:
    return list(map(lambda x: encode_sequence(x, tokenizer, split_chars), bunch.samples.values)), bunch.targets.values


def make_dataloader(data: list[BatchEncoding], targets: list[int]):
    tensor_data = TensorDataset(
        tensor(list(map(lambda x: x['input_ids'], data))),
        # tensor(list(map(lambda x: x['token_type_ids'], data))),
        tensor(list(map(lambda x: x['attention_mask'], data))),
        tensor(targets),
    )

    return DataLoader(tensor_data, batch_size=32, shuffle=True)
