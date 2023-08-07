from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin
from ..data.seq_bunch import SeqBunch


class ShallowLogisticModel(nn.Module):
    def __init__(self):
        super(ShallowLogisticModel, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class TorchModelWrapper:
    def create(self, train_data):
        pass

    def epoch(self):
        pass

    def get_model(self):
        pass


class TorchExperiment:
    def __init__(self, factory: TorchModelWrapper, test_data, train_data, encoding: TransformerMixin):
        pass

    def run(self):
        pass

# class TorchModelFactory(ABC):
#     def __init__(self):
#         self._model = None
#
#     @abstractmethod
#     def epoch(self):
#         pass
#
#     @abstractmethod
#     def finalize(self):
#         pass
#
#

#
#
# class LogisticTorchModel(TorchModel):
#
#     def create(self) -> nn.Module:
#         return ShallowLogisticModel()
#
#     def epoch(self):
#         pass
#
#     def finalize(self):
#         pass
#
#
# class TorchExperiment:
#     def __init__(self, model: nn.Module, dataset: SeqBunch, encoding: TransformerMixin):
#         self.model = model
#         self.dataset = TensorDataset(encoding.fit_transform(dataset), dataset.targets)
#
#     def epochs(self):
#         pass
#
#     def finalize(self):
#         pass
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# # Assuming 'binary.Encoder()' is a custom encoder, please import it if necessary
#
# # Assuming 'human_ds_train' contains the dataset, please replace it with your dataset
#
# # Encoding and preparing the data
# encoder = binary.Encoder()
# x = encoder.fit_transform(human_ds_train.samples)
# y = human_ds_train.targets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#
#
# # Create a PyTorch model
# class CustomModel(nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#         self.fc1 = nn.Linear(X_train_tensor.shape[1], 32)
#         self.fc2 = nn.Linear(32, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x
#
#
# model = CustomModel()
#
# # Define loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters())
#
#
# # Train the model
# def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=100, batch_size=32):
#     model.train()
#     dataset = TensorDataset(X_train, y_train)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels.unsqueeze(1))
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
#
#         # Add early stopping if needed based on validation set
#
#
# # Assuming you may want to use early stopping, use the below function instead of the previous one:
# # def train_model_early_stopping(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=32, patience=3):
# #     # Implement early stopping logic here, similar to Keras
# #     pass
#
# # Train the model
# train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)
# # If using early stopping, use the function train_model_early_stopping instead.
#
# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     y_pred = model(X_test_tensor)
#     y_pred = torch.round(y_pred).squeeze().numpy()
#     y_test_numpy = y_test_tensor.numpy()
#
# print(classification_report(y_test_numpy, y_pred))
