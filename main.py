from src.data.datasets import load_psi, Species
from src.features import anf

data = load_psi(Species.human)
encoder = anf.Encoder()

enc = encoder.fit_transform(data.samples)
print(enc)
