import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

SEED = 42

N_TRAIN = 6000
N_TEST = 2000

P_SIMPLE = 0.75

P_COMPLEX_GRID = np.array([
    0.75, 0.80, 0.85, 0.90, 0.95, 1.00
])

N_SEEDS = 3
EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.001

DEVICE = "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)