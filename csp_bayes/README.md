# Constraint Satisfaction and Bayesian Networks

This project studies the classic **map-coloring problem** through two
fundamentally different Artificial Intelligence paradigms:

1. **Deterministic reasoning** using Constraint Satisfaction Problems (CSPs)
2. **Probabilistic reasoning** using Bayesian Networks and Bayes’ Rule

The goal is to highlight how the same problem can be approached using
strict constraints versus uncertainty-aware inference.

---

## Problem Description

Given a map divided into regions, assign a color to each region such that
no two neighboring regions share the same color.

This problem is well-suited for comparing:
- Hard constraints with guaranteed correctness
- Probabilistic models that reason under uncertainty

---

## Part 1: Constraint Satisfaction Problem (CSP)

### Key Concepts
- Variables, domains, and constraints
- Binary constraints between neighboring regions
- Arc consistency
- AC-3 algorithm

### Approach

The map-coloring problem is formulated as a **Constraint Satisfaction Problem**:
- Each region is a variable
- Colors form the domain of each variable
- Adjacency relationships impose inequality constraints

The **AC-3 algorithm** is implemented to enforce arc consistency by iteratively
removing inconsistent values from variable domains until a valid coloring
is found or inconsistency is detected.

### Characteristics
- Deterministic and exact
- Guarantees correctness if a solution exists
- No notion of uncertainty or probability

---

## Part 2: Bayesian Network

### Key Concepts
- Conditional probability
- Bayes’ Rule
- Law of Total Probability
- Conditional independence
- Bayesian Network structure (DAG)
- Conditional Probability Tables (CPTs)

### Approach

A **Bayesian Network** is constructed to model probabilistic dependencies
between region colors. Each node represents a random variable corresponding
to a region’s color, and directed edges encode dependency assumptions.

The joint probability distribution is factorized according to the network
structure:

P(X₁, X₂, …, Xₙ) = ∏ P(Xᵢ | Parents(Xᵢ))


Custom **CPTs** are defined, and probabilities of specific color assignments
are computed using:
- Bayes’ Rule
- The Law of Total Probability
- Conditional independence assumptions implied by the graph structure

### Characteristics
- Probabilistic and uncertainty-aware
- Does not enforce hard constraints
- Allows reasoning even with incomplete or noisy information

---

## Conceptual Comparison

| Aspect | CSP | Bayesian Network |
|------|----|------------------|
| Reasoning type | Deterministic | Probabilistic |
| Handles uncertainty | No | Yes |
| Guarantees valid coloring | Yes | No |
| Uses Bayes’ Rule | No | Yes |
| Relies on independence assumptions | No | Yes |

---

## Key Takeaways

This project demonstrates:
- How **arc consistency** systematically reduces search space in CSPs
- How **Bayes’ Rule and conditional independence** enable compact
  representations of joint probability distributions
- Why Bayesian Networks can scale efficiently compared to full joint tables
- The trade-offs between exact constraint satisfaction and probabilistic inference

Together, these approaches illustrate two core pillars of Artificial
Intelligence: **logical constraint-based reasoning** and **probabilistic
reasoning under uncertainty**.
