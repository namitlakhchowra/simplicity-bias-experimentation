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

def complex_rule(x):
    
    return (np.sin(5 * np.pi * x) >= 0).astype(int)

def sample_simple_feature(rule_output, rng):
    """
    Generate feature values whose threshold-based
    predictions match the requested binary outputs.
    """

    x = np.empty(len(rule_output), dtype=np.float32)

    positive = rule_output == 1
    negative = rule_output == 0

    x[positive] = rng.uniform(
        0, 1, positive.sum()
    )

    x[negative] = rng.uniform(
        -1, 0, negative.sum()
    )

    return x
