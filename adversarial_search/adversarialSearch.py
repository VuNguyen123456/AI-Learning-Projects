# Basically minimax but with a graph structure instead of a tree
import heapq
from platform import node
# Global heuristic for A* and minimax utility
# heuristic = {
#         1: 7,
#         2: 4,
#         3: 6,
#         4: 1,
#         5: 2,
#         6: 0   # goal node => h=0
#     }
# Heuristic values for letter nodes
heuristic1 = {
    'A': 7,
    'B': 4,
    'C': 6,
    'D': 1,
    'E': 2,
    'F': 0   # goal node => h=0
}
heuristic2 = {
    'A': 6,
    'B': 5,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 0
}
heuristic3 = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 3,
    'E': 2,
    'F': 0
}

# Game class stays the same
class Game:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal

    def Actions(self, state):
        node, total_cost = state
        return [to for (to, _) in self.graph[node]]

    def Result(self, state, move):
        node, total_cost = state
        weight = next(w for (to, w) in self.graph[node] if to == move)

# Game:
class Game:
    def __init__(self, graph, start, goal):
        self.graph = graph    # adjacency list: {from: [(to, weight), ...]}
        self.start = start
        self.goal = goal

    # Always move max first so there's no need for this I think
    def ToMove(self, state):
        # You can alternate MAX and MIN based on depth (optional)
        # Here we leave turn logic for Minimax recursion
        pass

    def Actions(self, state):
        current_node, total_cost = state
        # Return neighbors you can legally move to
        return [to for (to, weight) in self.graph[current_node]]

    def Result(self, state, move):
        current_node, total_cost = state
        # Find the weight of the edge
        weight = next(w for (to, w) in self.graph[current_node] if to == move)
        # New state: move to next node, update total cost
        return (move, total_cost + weight) # total cost growing

    def IsTerminal(self, state):
        current_node, total_cost = state
        return current_node == self.goal

    def Utility(self, state, player):
        # Only called if terminal
        current_node, total_cost = state
        if player == 'MAX':
            return total_cost , None  # MAX wants max cost
        elif player == 'MIN':
            return -total_cost , None  # MIN wants to minimize MAX's gain

        
# Minimax
def minimaxSearch(game, state, h):
    # Ensure no cycle in graph traversal
    visited = set()
    def maxValue (game, state):
        # If done or already visited, return utility
        current_node, _ = state
        if game.IsTerminal(state) or current_node in visited:
            return game.Utility(state, player ='MAX')[0], [current_node]
        # Add to visited
        visited.add(current_node)
        v = float('-inf') # v, v2 are values of actions acting as bounds
        # Go through every possible next action and recursively find minValue (the min will find maxValue again)
        best_path = []
        for action in game.Actions(state):
            next_node, total_cost = game.Result(state, action) # next_node is the node we go to if we take this action
            v2, action2 = minValue(game, game.Result(state, action)) # v2 is value of that action (node, total cost)
            f_value = total_cost + h[next_node]
            if f_value > v:
                v, move = f_value, action
                best_path = [current_node] + action2 # Build path: action2 is path from next_node to goal
        visited.remove(current_node) # Backtrack so different branch can use this node later if they want
        return v, best_path

    def minValue (game, state):
        current_node, _ = state
        if game.IsTerminal(state) or current_node in visited:
            return game.Utility(state, player ='MIN')[0], [current_node]
        visited.add(current_node)
        v = float('inf')
        best_path = []
        for action in game.Actions(state):
            next_node, total_cost = game.Result(state, action)
            v2, action2 = maxValue(game, game.Result(state, action))
            f_value = total_cost + h[next_node]
            if f_value < v:
                v, move = f_value, action
                best_path = [current_node] + action2
        visited.remove(current_node)
        return v, best_path
    player = 'MAX'  # MAX always starts bascially game.toMove(state)
    value, move = maxValue(game, state)
    return move, value


def main():
    graph1 = {
        'A': [('B', 1), ('C', 3)],
        'B': [('D', 7), ('E', 3)],
        'C': [('E', 2)],
        'D': [('F', 1)],
        'E': [('F', 5)],
        'F': []  # goal node
    }
    game1 = Game(graph1, start='A', goal='F')
    start_state = ('A', 0)
    best_path, value = minimaxSearch(game1, start_state, heuristic1)
    print(f"Best path from start to goal: {best_path}")

    graph2 = {
        'A': [('B', 3), ('C', 4)],
        'B': [('D', 8), ('E', 6)],
        'C': [('E', 2)],
        'D': [('F', 5)],
        'E': [('F', 2)],
        'F': []   # Goal
    }


    game2 = Game(graph2, start='A', goal='F')
    start_state2 = ('A', 0) # (node, total_cost)
    best_path2, value2 = minimaxSearch(game2, start_state2, heuristic2)
    print(f"Best path from start to goal: {best_path2}")

    graph3 = {
        'A': [('B', 2), ('C', 5)],
        'B': [('D', 2), ('E', 4)],
        'C': [('E', 1), ('B', 3)],
        'D': [('F', 2)],
        'E': [('D', 1), ('F', 3)],
        'F': []
    }


    game3 = Game(graph3, start='A', goal='F')
    start_state3 = ('A', 0) # (node, total_cost)
    best_path3, value3 = minimaxSearch(game3, start_state3, heuristic3)
    print(f"Best path from start to goal: {best_path3}")

if __name__ == "__main__":
    main()
