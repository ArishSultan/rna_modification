import torch
import numpy

SEED_VALUE = 42


def reset_random():
    numpy.random.seed(SEED_VALUE)

    torch.manual_seed(SEED_VALUE)
    torch.mps.manual_seed(SEED_VALUE)
