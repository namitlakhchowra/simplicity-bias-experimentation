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

def sample_complex_feature(rule_output, rng):
    """
    Generate feature values whose sine-based predictions
    match the requested binary outputs.
    """


    x = np.empty(
        len(rule_output),
        dtype=np.float32
    )

    remaining = np.ones(
        len(rule_output),
        dtype=bool
    )

    while remaining.any():

        indices = np.flatnonzero(remaining)

        candidates = rng.uniform(
            -1, 1, len(indices)
        )

        predictions = complex_rule(candidates)

        accepted = (
            predictions == rule_output[indices]
        )

        accepted_indices = indices[accepted]

        x[accepted_indices] = candidates[accepted]

        remaining[accepted_indices] = False

    return x

def generate_data(
    n_samples,
    p_simple,
    p_complex,
    rng
):
    """
    Generate two features and a shared binary target.

    p_simple:
        Probability that the simple feature's rule
        agrees with the target.

    p_complex:
        Probability that the complex feature's rule
        agrees with the target.
    """

    y = rng.integers(
        0, 2, n_samples
    )

    simple_correct = (
        rng.random(n_samples) < p_simple
    )

    complex_correct = (
        rng.random(n_samples) < p_complex
    )

    b_simple = np.where(
        simple_correct,
        y,
        1 - y
    )

    b_complex = np.where(
        complex_correct,
        y,
        1 - y
    )

    x_simple = sample_simple_feature(
        b_simple, rng
    )

    x_complex = sample_complex_feature(
        b_complex, rng
    )

    X = np.column_stack([
        x_simple,
        x_complex
    ]).astype(np.float32)

    return (
        X,
        y.astype(np.int64),
        b_simple,
        b_complex
    )
