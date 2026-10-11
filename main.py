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

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

def train_model(X, y, seed):

    torch.manual_seed(seed)

    model = NeuralNetwork().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
        device=DEVICE
    )

    dataset_size = len(X)

    for epoch in range(EPOCHS):

        model.train()

        permutation = torch.randperm(
            dataset_size
        )

        for start in range(
            0, dataset_size, BATCH_SIZE
        ):

            indices = permutation[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            batch_X = X_tensor[indices]
            batch_y = y_tensor[indices]

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = loss_function(
                logits, batch_y
            )

            loss.backward()
            optimizer.step()

    return model

def predict(model, X):

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE
        )

        logits = model(X_tensor)

        predictions = logits.argmax(
            dim=1
        )

    return predictions.cpu().numpy()

def generate_conflict_set(n_samples, rng):
    """
    Generate examples where the simple and complex
    feature rules always disagree.

    The simple feature predicts b_simple.
    The complex feature predicts 1 - b_simple.
    """

    b_simple = rng.integers(
        0, 2, n_samples
    )

    b_complex = 1 - b_simple

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

    return X, b_simple

def evaluate_model(
    model,
    p_simple,
    p_complex,
    seed
):

    rng = np.random.default_rng(seed)

    # Evaluate ordinary test accuracy
    X_test, y_test, _, _ = generate_data(
        N_TEST,
        p_simple,
        p_complex,
        rng
    )

    predictions = predict(
        model, X_test
    )

    test_accuracy = np.mean(
        predictions == y_test
    )

    # Evaluate reliance on the conflict set
    X_conflict, b_simple = generate_conflict_set(
        N_TEST,
        rng
    )

    conflict_predictions = predict(
        model, X_conflict
    )

    simple_reliance = np.mean(
        conflict_predictions == b_simple
    )

    return test_accuracy, simple_reliance

results = []

for p_complex in P_COMPLEX_GRID:

    accuracies = []
    reliances = []

    print(
        f"\nComplex predictiveness: {p_complex:.2f}"
    )

    for seed in range(N_SEEDS):

        rng = np.random.default_rng(
            SEED + seed
        )

        # Generate training data
        X_train, y_train, _, _ = generate_data(
            N_TRAIN,
            P_SIMPLE,
            p_complex,
            rng
        )

        # Train model
        model = train_model(
            X_train,
            y_train,
            SEED + seed
        )

        # Evaluate model
        accuracy, reliance = evaluate_model(
            model,
            P_SIMPLE,
            p_complex,
            SEED + 1000 + seed
        )

        accuracies.append(accuracy)
        reliances.append(reliance)

        print(
            f"Seed {seed + 1}: "
            f"accuracy={accuracy:.3f}, "
            f"simple reliance={reliance:.3f}"
        )

    results.append({
        "p_complex": p_complex,
        "test_accuracy": np.mean(accuracies),
        "test_accuracy_std": np.std(accuracies),
        "simple_reliance": np.mean(reliances),
        "reliance_std": np.std(reliances)
    })

print("\n========== FINAL RESULTS ==========\n")

print(
    f"{'P Complex':>12} "
    f"{'Test Accuracy':>15} "
    f"{'Simple Reliance':>17}"
)

for result in results:

    print(
        f"{result['p_complex']:>12.2f} "
        f"{result['test_accuracy']:>15.3f} "
        f"{result['simple_reliance']:>17.3f}"
    )