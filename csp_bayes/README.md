# Constraint Satisfaction and Bayesian Networks

This project addresses the classic map-coloring problem using two AI
perspectives: constraint satisfaction and probabilistic modeling.

The task is to assign colors to regions such that no neighboring regions share
the same color.

---

## Part 1: Constraint Satisfaction Problem (CSP)

### Concepts
- Variables, domains, and constraints
- Arc consistency
- AC-3 algorithm

### Approach
The map-coloring problem is formulated as a CSP. The AC-3 algorithm is implemented
to enforce arc consistency and reduce domains until a valid coloring is found.

---

## Part 2: Bayesian Network

### Concepts
- Bayesian Network structure
- Conditional Probability Tables (CPTs)
- Joint probability computation

### Approach
A Bayesian Network is constructed to model dependencies between region colors.
Custom CPTs are defined, and joint probabilities for specific color assignments
are computed and verified.

---

## Focus
This project highlights the contrast between:
- Deterministic constraint-based reasoning
- Probabilistic reasoning under uncertainty
