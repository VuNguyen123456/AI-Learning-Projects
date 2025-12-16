# Iterative Deepening Search on Graphs

This project implements an Iterative Deepening Search (IDS) algorithm designed
to operate on graphs that may contain cycles, without falling into infinite
loops.

The goal is to search for a node whose numeric value satisfies a specific
property (being a prime number).

---

## Key Concepts
- Depth-limited search
- Iterative deepening strategy
- Cycle detection in graphs
- Goal testing based on numeric properties

---

## Approach
The algorithm performs repeated depth-limited searches, gradually increasing
the depth limit until a solution is found or all reachable nodes are exhausted.
A visited-state mechanism is used to prevent infinite traversal in cyclic
graphs.

---

## Demonstrations
The implementation is tested on:
- A cyclic graph with no valid solution
- A graph with a single valid solution
- A graph with multiple valid solutions

Each case demonstrates correctness and termination.

---

## Analysis
The project includes discussion and analysis of:
- Best-case time complexity
- Worst-case time complexity
- Trade-offs between depth-first and breadth-first approaches
