def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to √n
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Graph is a adjacency list
def IDS(G, start, max_depth):
    for depth in range(max_depth + 1):
        path = [start]
        result = DBS(G, start, depth, path)
        if result:
            return result
    return None

def DBS(G, node, remain_depth, curPath):
    if is_prime(node):
        return curPath
    if remain_depth == 0:
        return None
    for child in G[node]:
        if child not in curPath: # No cycle
            curPath.append(child)
            result = DBS(G, child, remain_depth - 1, curPath)
            if result:
                return result
            curPath.pop()
    return None

def main():
    # A graph with cycles, no solution
    # Cycles: 4->6->8->4 and 6->9->12->15
    graph1 = {
        4: [6, 8],     
        6: [8, 9],      
        8: [4, 9],      
        9: [6, 12],  
        12: [15],     
        15: []
    }
    
    # A graph with one solution
    # The only prime is 23
    graph2 = {
        10: [12, 14],   
        12: [15, 16],   
        14: [18],       
        15: [21], 
        16: [20],      
        18: [22],       
        20: [23],       
        21: [25],       
        22: [24],      
        23: [],         
        24: [],         
        25: []          
    }

    # A graph with multiple solutions
    # The primes are 11, 13, and 17
    graph3 = {
        1: [4, 6],
        4: [8, 11],     
        6: [9, 13],     
        8: [15],       
        9: [17],        
        11: [],        
        13: [],        
        15: [],         
        17: []          
    }
    
    print("Graph 1 (No solution):", IDS(graph1, 4, 5))  # Expect None
    print("Graph 2 (One solution):", IDS(graph2, 10, 5)) # Expect a path leading to 23
    print("Graph 3 (Multiple solutions):", IDS(graph3, 1, 5)) # Expect a path leading to 11, 13, or 17
 
if __name__ == "__main__":
    main()
