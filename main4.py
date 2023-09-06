from src.data import load_psi, Species
from src.features.encodings import pstnpss

encoder = pstnpss.Encoder()

human_data = encoder.fit_transform(load_psi(Species.human))
yeast_data = encoder.fit_transform(load_psi(Species.yeast))

print(human_data.samples)
print(yeast_data.samples)
