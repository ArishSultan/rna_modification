from tqdm.notebook import tqdm
from transformers import PreTrainedModel

from torch import no_grad
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import softmax


def _move_batch_to_device(device: str, batch):
    return [b.to(device) for b in batch]


def _get_model_outputs(batch, model: PreTrainedModel):
    outputs = model(
        input_ids=batch[0],
        # token_type_ids=batch[1],
        attention_mask=batch[1],
        labels=batch[2],
    )

    return outputs


def _get_model_acc(outputs, labels, average=False):
    predictions = softmax(outputs.logits, dim=1).argmax(dim=1)
    correct_count = (predictions == labels).sum().item()

    if average:
        return correct_count / len(labels)

    return correct_count


def calculate_acc_dataset(device: str, model: PreTrainedModel, dataloader: DataLoader):
    accuracy = 0
    for batch in dataloader:
        batch = _move_batch_to_device(device, batch)

        outputs = _get_model_outputs(batch, model)
        accuracy += _get_model_acc(outputs, batch[2])
    return accuracy / len(dataloader.dataset)


def train_epoch(
        epoch: int,
        device: str,
        model: PreTrainedModel,
        optimizer: Optimizer,
        train_data: DataLoader,
        val_data: DataLoader = None
):
    with tqdm(range(len(train_data))) as bar:
        bar.set_description(f'Epoch {epoch}')

        for batch in train_data:
            batch = _move_batch_to_device(device, batch)
            outputs = _get_model_outputs(batch, model)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            bar.set_postfix({'train_acc': _get_model_acc(outputs, batch[2], True)})
            bar.update()

        model.eval()
        with no_grad():
            train_acc = calculate_acc_dataset(device, model, train_data)
            postfix_data = {'train_acc': train_acc}

            if val_data is not None:
                val_acc = calculate_acc_dataset(device, model, val_data)
                postfix_data['val_acc'] = val_acc
            else:
                val_acc = None

            bar.set_postfix(postfix_data, refresh=True)

        return train_acc, val_acc
