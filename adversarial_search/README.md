# Adversarial Game Search

This project explores adversarial decision-making in Artificial Intelligence
using deterministic, two-player, zero-sum games with perfect information. The
focus is on how rational agents reason and plan when their objectives directly
conflict.

---

## Problem Setting

The environment is modeled as a directed graph with a designated start node and a
goal node. Each edge has an associated cost representing the value accumulated
when traversing that edge.

Two players alternate turns:
- **MAX** attempts to maximize the total path cost
- **MIN** attempts to minimize the total path cost

The game terminates when the goal node is reached, at which point a utility value
is assigned.

---

## Game Tree Representation

The graph is transformed into a game tree where:
- Each node represents a game state
- Edges represent legal actions
- Tree depth corresponds to turn order

Player turns alternate by depth:
- MAX acts at even depths
- MIN acts at odd depths

Terminal nodes correspond to reaching the goal state, where utility values are
computed and propagated upward.

---

## Minimax Algorithm

The **minimax algorithm** computes an optimal strategy under the assumption that
both players act rationally and optimally.

- MAX selects actions that maximize utility values
- MIN selects actions that minimize utility values

The algorithm recursively explores the game tree, evaluating terminal states and
backing up values to determine the best action at the root.

### Properties

- Guarantees optimal play
- Explores the full game tree
- Time complexity: O(b^m)
- Space complexity: O(m)

where *b* is the branching factor and *m* is the maximum depth of the game tree.

---

## Alpha-Beta Pruning

Alpha-beta pruning improves minimax efficiency by eliminating branches that
cannot affect the final decision.

Two bounds are maintained:
- **α (alpha):** the best value found so far for MAX
- **β (beta):** the best value found so far for MIN

A branch is pruned when:

β ≤ α

Alpha-beta pruning returns the same optimal result as minimax while exploring
significantly fewer nodes in practice.

---

## Heuristic-Guided Play

To improve scalability in larger graphs, heuristic strategies are incorporated
to guide player decisions without exhaustively exploring the entire game tree.

### MIN Strategy: A* Search

MIN follows an **A\*** strategy and selects moves that minimize:

f(n) = g(n) + h(n)

where:
- g(n) is the accumulated path cost
- h(n) is an admissible heuristic estimating remaining cost

This allows MIN to efficiently approximate optimal play.

### MAX Strategy: Reverse A* Search

MAX follows a **reverse A\*** strategy and selects moves that maximize:

f(n) = g(n) + h(n)

This models adversarial behavior where MAX actively seeks higher-cost paths.

---

## Comparison of Approaches

| Method | Optimal | Full Tree Exploration | Efficiency |
|------|--------|----------------------|------------|
| Minimax | Yes | Yes | Low |
| Alpha-Beta Pruning | Yes | No | High |
| Heuristic-Guided Play | Approximate | No | Very High |

---

## Key Takeaways

This project demonstrates:
- How adversarial problems are modeled as game trees
- Why minimax guarantees optimal decisions under rational play
- How alpha-beta pruning reduces exponential search complexity
- The trade-off between optimality and efficiency when heuristics are used

These techniques form the foundation of AI systems used in game-playing,
planning, and competitive decision-making.
