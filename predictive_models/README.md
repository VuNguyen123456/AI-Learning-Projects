# Predictive Modeling: Decision Trees and Neural Networks

This project explores **predictive modeling** using a synthetic restaurant waiting dataset. The goal is to understand and compare the behavior of **symbolic AI** (decision trees) versus **neural networks** for a simple binary classification problem.

---

## Project Overview

- **Task:** Predict whether a customer will wait for a table at a restaurant (`WillWait`).
- **Dataset:** Synthetic, rule-based, 100 examples, balanced class distribution.
- **Learning Paradigms:** 
  - **Decision Tree:** Symbolic, rule-based classification.
  - **Neural Network:** Feedforward, probabilistic classification with thresholding.
- **Evaluation:** Accuracy on held-out test data, with experiments across varying training set sizes (20%, 30%, 40%).

---

## Data Generation

- Each example consists of 10 discrete features:
  - Boolean: `Alternate`, `Bar`, `Fri/Sat`, `Hungry`, `Raining`, `Reservation`
  - Categorical: `Patrons`, `Price`, `Type`, `WaitEstimate`
- **Labeling Logic:** Determined by a hand-crafted decision tree.
- Balanced dataset ensures 50/50 split between `WillWait = True` and `False`.

---

## Models Implemented

### Decision Tree

- Custom recursive implementation following the **Learn-Decision-Tree algorithm**.
- Selects the **most important attribute** at each node based on purity.
- Handles base cases:
  1. No examples → majority of parent examples.
  2. All examples same label → leaf node.
  3. No attributes left → majority label of current examples.
- **Tree Representation:** Nested dictionaries mapping attribute values to subtrees or leaf labels.

### Neural Network

- Simple feedforward neural network:
  - Input layer → Hidden layer (tanh activation) → Output layer (sigmoid activation).
  - Supports one-hot encoding for categorical features.
- Training via gradient descent with binary cross-entropy loss.
- Outputs probabilities, classified with threshold 0.5.

---

## Evaluation

- Train/test splits created at ratios of 20%, 30%, 40%.
- Accuracy is computed on both training and test sets.
- Sample predictions are displayed to observe model behavior.

**Focus:**
- Compare **symbolic AI** vs **neural network** approaches.
- Analyze performance with **limited training data**.
- Understand generalization behavior on unseen examples.

---

## Getting Started

### Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `collections` (standard library)

### Usage

```bash
# Run the main script
python restaurant_predictive_model.py
