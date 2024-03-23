# from src.model import xgboost
# from src.utils import write_reports
# from src.experiment import Experiment
# from src.features.encodings import pstnpss
# from src.dataset import load_dataset, Species, Modification
#
# psi_encoder = pstnpss.Encoder()
# human_dataset = load_dataset(Species.human, Modification.psi)
#
# experiment = Experiment(xgboost.Factory(), None, human_dataset, psi_encoder, k=10)
# report = experiment.run()
#
# write_reports(report, 'PSI PSTNPSS Ensemble', Modification.psi.value, Species.human.value)
#
#
# m6a_encoder = pstnpss.Encoder()
# human_dataset = load_dataset(Species.human, Modification.m6a)
#
# experiment = Experiment(xgboost.Factory(), None, human_dataset, m6a_encoder, k=10)
# report = experiment.run()
#
# write_reports(report, 'M6A PSTNPSS Ensemble', Modification.m6a.value, Species.human.value)


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from src.dataset import load_dataset, Species, Modification

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Load your dataset
train_data = load_dataset(Species.human, Modification.psi)


# A custom dataset class in PyTorch
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Preprocess and encode the texts
train_encodings = tokenizer([value[0] for value in train_data.samples.values], truncation=True, padding=True,
                            max_length=41, return_tensors="pt")

# Initialize our custom dataset
train_dataset = CustomDataset(train_encodings, train_data.targets)

# Load BERT model for sequence classification
model_bert = BertForSequenceClassification.from_pretrained('bert-base-cased')
model_bert.to('mps')

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Prepare optimizer and loss function
optimizer = torch.optim.Adam(model_bert.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# Function to train the model
def train(epoch, model, loader, optimizer, loss_fn):
    model.train()
    for _, data in enumerate(loader):
        inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
        labels = data['labels'].to(model.device)

        # Forward pass
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}], Loss: {loss.item()}")

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    train(epoch, model_bert, train_loader, optimizer, loss_fn)
