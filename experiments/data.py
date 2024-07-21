from pandas import concat
from src.model.porpoise import pstnpss
from src.dataset import load_benchmark_dataset, Species, Modification
from src.features.encodings import multiple, binary, ncp, pse_knc

encoder = multiple.Encoder([
    binary.Encoder(),
    ncp.Encoder(),
    pstnpss.Encoder(Species.human),
    pse_knc.Encoder()
])

train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

samples = encoder.fit_transform(train_dataset.samples, y=train_dataset.targets)

concat([train_dataset.targets, samples], axis=1).to_csv('xxx2.csv', index=False)
