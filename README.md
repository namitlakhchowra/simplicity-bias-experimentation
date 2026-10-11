# Can Simplicity Override Predictive Strength?

This project investigates simplicity bias in neural networks, examining whether models prefer simpler features even when more complex features are more predictive.

A synthetic binary classification dataset is generated using a threshold-based simple feature and a nonlinear sine-based complex feature. The simple feature's predictive strength remains fixed at 75%, while the complex feature's predictive strength increases from 75% to 100%.

A neural network is trained and evaluated across different settings. Feature reliance is measured using a conflict set where both features disagree. The experiment estimates the transition point and visualizes how predictive strength influences feature preference.
