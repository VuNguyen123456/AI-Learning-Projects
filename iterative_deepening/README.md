# Iterative Deepening Search (IDS) on Graphs with Cycles

This project implements Iterative Deepening Search (IDS), a graph search
algorithm that combines the completeness of Breadth-First Search (BFS)
with the space efficiency of Depth-First Search (DFS).

The algorithm searches for a goal node whose numeric value is a prime
number, while safely handling graphs that contain cycles.

---

## Background: Why Iterative Deepening?

Classical search strategies have important trade-offs:

- **Breadth-First Search (BFS)** guarantees finding a shortest solution
  but requires exponential space.
- **Depth-First Search (DFS)** is space-efficient but may fail to find
  a solution or enter infinite loops on cyclic graphs.

**Iterative Deepening Search (IDS)** combines the strengths of both:
- Performs repeated depth-limited DFS
- Gradually increases the depth bound
- Finds a solution with the fewest number of edges (if one exists)

---

## Algorithm Overview

IDS works by repeatedly invoking a **Depth-Bounded Search (DBS)**:

1. Start with depth bound = 0 and explore only the start node.
2. Increment the depth bound (1, 2, 3, …).
3. At each iteration, perform a depth-first search that does not exceed
   the current depth bound.
4. Stop when a goal node is found or the search space is exhausted.

To ensure termination on graphs with cycles, this implementation tracks
the current search path and prevents revisiting nodes already on the path.

---

## Handling Cycles and Termination

Pure IDS does **not** inherently protect against cycles. On cyclic graphs,
this can lead to infinite depth expansion.

This implementation explicitly avoids cycles by:
- Tracking the current path during DFS
- Preventing revisiting nodes already in the path

This guarantees termination for finite graphs while preserving the
properties of IDS.

---

## Implementation Details

### Prime Number Goal Test
A helper function checks whether a node value is prime:

```python
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
