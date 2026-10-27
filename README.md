# Can Simplicity Override Predictive Strength?

A small PyTorch experiment testing whether neural networks prefer a simple feature over a more predictive complex one, and determining the threshold.

## Features / Overview

- Generates a synthetic binary classification dataset with two features: a simple threshold feature and a complex sine-based feature
- Holds the simple feature's predictive strength fixed at 75% and sweeps the complex feature's from 75% to 100%
- Trains a small MLP across 3 seeds at each setting
- Measures feature reliance on a conflict set where the two features always disagree
- Estimates the transition point where the network switches to the complex feature, and the resulting "simplicity premium"
- Saves results to `results.csv` and a plot to `simplicity_bias_results.png`

**Problem it addresses:** neural networks tend to latch onto simple features even when more complex ones predict the label better. This project quantifies how much more predictive the complex feature must be before that bias is overcome.

## Installation

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/namitlakhchowra/simplicity-bias-experimentation.git
cd simplicity-bias-experimentation
pip install numpy pandas matplotlib torch
python main.py
```



