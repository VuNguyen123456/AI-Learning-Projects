###############################CSP and AC-3 Algorithm Implementation###############################
class CSP:
    def __init__(self, variables, domains, arcs, neighbors):
        self.variables = variables
        self.domains = domains
        self.arcs = arcs
        self.neighbors = neighbors

    def is_consistent(self, var1, value1, var2, value2):
        """The constraint: neighbors must have different colors"""
        if var2 not in self.neighbors[var1]: # If they are not neighbors, no constraint
            return True
        return value1 != value2

def ac3(csp):
    arc_set = set(csp.arcs)
    while arc_set:
        X, Y = arc_set.pop()
        if revise(csp, X, Y):
            if not csp.domains[X]: # if it's empty
                return False
            # When you remove a value from X it might affect X's neighbors so need to recheck
            for neighbor in csp.neighbors[X]: # recheck all neighbors
                if neighbor != Y: # avoid Y because we just check this
                    arc_set.add((neighbor, X))
    return True

def revise(csp, X, Y):
    ret = False
    domain_x = csp.domains[X].copy() # Copy because we might remove values during iteration
    for x in domain_x: # Loop thorugh domain of x
        satis = False
        for y in csp.domains[Y]: # Loop through domain of y to see if there are any constraint satisfied between them
            if csp.is_consistent(X, x, Y, y): # if satisfied meaning that don't remove this
                satis = True
                break
        if not satis:
            csp.domains[X].remove(x)
            ret = True
    return ret


################################### Bayesian Network Implementation ###################################
# 1. Need to network structures (Who are the parent)
# 2. Store CPTs: With all the probability designed
# 3. Compute Joint Probability : a function that's able to calculate probability given the stuff.
cpts = {
        'WA':{
            'red': 1/4, 
            'green': 1/4, 
            'blue': 1/4, 
            'yellow': 1/4
        },
        'NT':{
            # Given WA
            'red': { 'red': 0, 'green': 1/3, 'blue': 1/3, 'yellow': 1/3 },
            'green': { 'red': 1/3, 'green': 0, 'blue': 1/3, 'yellow': 1/3 },
            'blue': { 'red': 1/3, 'green': 1/3, 'blue': 0, 'yellow': 1/3 },
            'yellow': { 'red': 1/3, 'green': 1/3, 'blue': 1/3, 'yellow': 0 }
        },
        'SA':{
            # Given WA and NT: (WA, NT)
            ('red', 'red'): {'red': 0, 'green': 1/3, 'blue': 1/3, 'yellow': 1/3},
            ('red', 'green'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('red', 'blue'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('red', 'yellow'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('green', 'red'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('green', 'green'): {'red': 1/3, 'green': 0, 'blue': 1/3, 'yellow': 1/3},
            ('green', 'blue'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('green', 'yellow'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('blue', 'red'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('blue', 'green'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('blue', 'blue'): {'red': 1/3, 'green': 1/3, 'blue': 0, 'yellow': 1/3},
            ('blue', 'yellow'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'red'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'green'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'blue'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'yellow'): {'red': 1/3, 'green': 1/3, 'blue': 1/3, 'yellow': 0}
        },
        'Q': {
            # Given NT and SA: (SA,NT)
            ('red', 'red'): {'red': 0, 'green': 1/3, 'blue': 1/3, 'yellow': 1/3},
            ('red', 'green'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('red', 'blue'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('red', 'yellow'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('green', 'red'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('green', 'green'): {'red': 1/3, 'green': 0, 'blue': 1/3, 'yellow': 1/3},
            ('green', 'blue'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('green', 'yellow'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('blue', 'red'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('blue', 'green'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('blue', 'blue'): {'red': 1/3, 'green': 1/3, 'blue': 0, 'yellow': 1/3},
            ('blue', 'yellow'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'red'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'green'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'blue'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'yellow'): {'red': 1/3, 'green': 1/3, 'blue': 1/3, 'yellow': 0}
        },
        'NSW': {
            # Given Q and SA: (Q, SA)
            ('red', 'red'): {'red': 0, 'green': 1/3, 'blue': 1/3, 'yellow': 1/3},
            ('red', 'green'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('red', 'blue'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('red', 'yellow'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('green', 'red'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('green', 'green'): {'red': 1/3, 'green': 0, 'blue': 1/3, 'yellow': 1/3},
            ('green', 'blue'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('green', 'yellow'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('blue', 'red'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('blue', 'green'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('blue', 'blue'): {'red': 1/3, 'green': 1/3, 'blue': 0, 'yellow': 1/3},
            ('blue', 'yellow'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'red'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'green'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'blue'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'yellow'): {'red': 1/3, 'green': 1/3, 'blue': 1/3, 'yellow': 0}
        },
        'V': {
            # Given SA and NSW: (SA, NSW)
            ('red', 'red'): {'red': 0, 'green': 1/3, 'blue': 1/3, 'yellow': 1/3},
            ('red', 'green'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('red', 'blue'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('red', 'yellow'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('green', 'red'): {'red': 0, 'green': 0, 'blue': 1/2, 'yellow': 1/2},
            ('green', 'green'): {'red': 1/3, 'green': 0, 'blue': 1/3, 'yellow': 1/3},
            ('green', 'blue'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('green', 'yellow'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('blue', 'red'): {'red': 0, 'green': 1/2, 'blue': 0, 'yellow': 1/2},
            ('blue', 'green'): {'red': 1/2, 'green': 0, 'blue': 0, 'yellow': 1/2},
            ('blue', 'blue'): {'red': 1/3, 'green': 1/3, 'blue': 0, 'yellow': 1/3},
            ('blue', 'yellow'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'red'): {'red': 0, 'green': 1/2, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'green'): {'red': 1/2, 'green': 0, 'blue': 1/2, 'yellow': 0},
            ('yellow', 'blue'): {'red': 1/2, 'green': 1/2, 'blue': 0, 'yellow': 0},
            ('yellow', 'yellow'): {'red': 1/3, 'green': 1/3, 'blue': 1/3, 'yellow': 0}
        },
        'T': {
            # T has no neighbors, so uniform distribution
            'red': 0.25,
            'green': 0.25,
            'blue': 0.25,
            'yellow': 0.25
        }
}


class BayesianNetwork:
    def __init__(self, variables, parents, cpts):
        self.variables = variables # The nodes and their structures, list of all nodes
        self.parents = parents # The edges associate with the nodes?, a dictionary mapping each node to their parents
        self. cpts = cpts # The CPTs probability that goes with each nodes, each cpt is a dic (nested?) or something

    def calcPrAustralianMap(self, joint_probability):
        WA = cpts['WA'][joint_probability['WA']]
        NT = cpts['NT'][joint_probability['WA']][joint_probability['NT']]
        SA = cpts['SA'][(joint_probability['WA'], joint_probability['NT'])][joint_probability['SA']]
        Q = cpts['Q'][(joint_probability['SA'], joint_probability['NT'])][joint_probability['Q']]
        NSW = cpts['NSW'][(joint_probability['Q'], joint_probability['SA'])][joint_probability['NSW']]
        V = cpts['V'][(joint_probability['SA'], joint_probability['NSW'])][joint_probability['V']]
        T = cpts['T'][joint_probability['T']]
        return WA * NT * SA * Q * NSW * V * T

#  Compute and verify the joint probability:
#   𝑃𝑟[𝑊𝐴 = 𝑟𝑒𝑑, 𝑁𝑇 = 𝑏𝑙𝑢𝑒, 𝑆𝐴 = 𝑔𝑟𝑒𝑒𝑛, 𝑄 = 𝑦𝑒𝑙𝑙𝑜𝑤, 𝑁𝑆𝑊 = 𝑟𝑒𝑑, 𝑉 = 𝑏𝑙𝑢𝑒,𝑇 = 𝑦𝑒𝑙𝑙𝑜𝑤]
#  so the function take in a dictionary of variable assignments and return the joint probability


def main():
    ################################################################ CSP ############################################
    # Var
    v1 = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    # Domains before arc consistency
    d1 = {
        'WA': ['red', 'green', 'blue', 'yellow'],
        'NT': ['red', 'green', 'blue', 'yellow'],
        'SA': ['red', 'green', 'blue', 'yellow'],
        'Q': ['red', 'green', 'blue', 'yellow'],
        'NSW': ['red', 'green', 'blue', 'yellow'],
        'V': ['red', 'green', 'blue', 'yellow'],
        'T': ['red', 'green', 'blue', 'yellow']
    }
    # Arcs
    a1 = [
        ('WA', 'NT'), ('NT', 'WA'),
        ('WA', 'SA'), ('SA', 'WA'),
        ('NT', 'SA'), ('SA', 'NT'),
        ('NT', 'Q'), ('Q', 'NT'),
        ('SA', 'Q'), ('Q', 'SA'),
        ('SA', 'NSW'), ('NSW', 'SA'),
        ('SA', 'V'), ('V', 'SA'),
        ('Q', 'NSW'), ('NSW', 'Q'),
        ('NSW', 'V'), ('V', 'NSW')
    ]

    # neighbors
    n1 = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['SA', 'Q', 'V'],
        'V': ['SA', 'NSW'],
        'T': []
    }
    
    csp = CSP(v1, d1, a1, n1)

    # # Var
    # v1 = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    # # Domains before arc consistency
    # d1 = {
    #     'WA': ['red'],
    #     'NT': ['red', 'green', 'blue', 'yellow'],
    #     'SA': ['red', 'green', 'blue', 'yellow'],
    #     'Q': ['red', 'green', 'blue', 'yellow'],
    #     'NSW': ['red', 'green', 'blue', 'yellow'],
    #     'V': ['red', 'green', 'blue', 'yellow'],
    #     'T': ['red', 'green', 'blue', 'yellow']
    # }
    # # Arcs
    # a1 = [
    #     ('WA', 'NT'), ('NT', 'WA'),
    #     ('WA', 'SA'), ('SA', 'WA'),
    #     ('NT', 'SA'), ('SA', 'NT'),
    #     ('NT', 'Q'), ('Q', 'NT'),
    #     ('SA', 'Q'), ('Q', 'SA'),
    #     ('SA', 'NSW'), ('NSW', 'SA'),
    #     ('SA', 'V'), ('V', 'SA'),
    #     ('Q', 'NSW'), ('NSW', 'Q'),
    #     ('NSW', 'V'), ('V', 'NSW')
    # ]

    # # neighbors
    # n1 = {
    #     'WA': ['NT', 'SA'],
    #     'NT': ['WA', 'SA', 'Q'],
    #     'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    #     'Q': ['NT', 'SA', 'NSW'],
    #     'NSW': ['SA', 'Q', 'V'],
    #     'V': ['SA', 'NSW'],
    #     'T': []
    # }
    # csp = CSP(v1, d1, a1, n1)
    # Test AC-3 Example 1
    print("Domains before AC-3:")
    for var in csp.variables:
        print(f"{var}: {csp.domains[var]}")
    ac3(csp)
    print("\nDomains after AC-3:")
    for var in csp.variables:
        print(f"{var}: {csp.domains[var]}")

    ################################################################ Bayesian ############################################
    # Var
    v1 = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    # parents
    p1 = {
        'WA': [],
        'NT': ['WA'],
        'SA': ['WA', 'NT'],
        'Q': ['SA', 'NT'],
        'NSW': ['SA', 'Q'],
        'V': ['SA', 'NSW'],
        'T': []
    }
    # cpt
    c1 = cpts
    bn = BayesianNetwork(v1, p1, c1)
    pr = {'WA': 'red', 'NT': 'blue', 'SA': 'green', 'Q': 'yellow', 'NSW': 'red', 'V': 'blue', 'T': 'yellow'}
    print(bn.calcPrAustralianMap(pr))

if __name__ == "__main__":
    main()
