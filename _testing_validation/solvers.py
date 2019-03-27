from src import *

def pso(n, my_func, bounds, dimension, max_nfe, w, c1, c2):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    [Xi.getFitness() for Xi in X]
    
    while Solution.nfe < max_nfe:
        #Round 1
        S  = op.select_current(X)
        U  = op.w_pso(S, w, c1, c2)
        X  = U
        
        [Xi.getFitness() for Xi in X]
    return Solution

def de(n, my_func, bounds, dimension, max_nfe, beta, pr):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    [Xi.getFitness() for Xi in X]
    
    while Solution.nfe < max_nfe:
        #Round 1
        S1 = op.select_random(X, 1)
        S2 = op.select_random(X, 1)
        S3 = op.select_random(X, 1)
        U  = op.w_mut_de(S1, S2, S3, beta)
        #Round 2
        S1 = op.select_current(X)
        S2 = op.select_current(U)
        U  = op.w_crx_exp2(S1, S2, pr)
        X  = op.replace_if_best(X, U)
        
        [Xi.getFitness() for Xi in X]
    return Solution

def ga(n, my_func, bounds, dimension, max_nfe, k, alpha, pr):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    [Xi.getFitness() for Xi in X]
    
    while Solution.nfe < max_nfe:
        #Round 1
        S1 = op.select_tournament(X, n=1, k=int(k))
        S2 = op.select_tournament(X, n=1, k=int(k))
        U  = op.w_crx_blend2(S1, S2, alpha)
        X  = op.replace_if_best(X, U)
        #Round 2
        S  = op.select_current(X)
        U  = op.w_mut_uni(S, pr)
        X  = U
        
        [Xi.getFitness() for Xi in X]
    return Solution

def cs(n, my_func, bounds, dimension, max_nfe, pr, k):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    [Xi.getFitness() for Xi in X]
    
    while Solution.nfe < max_nfe:
        #Round 1
        S  = op.select_current(X)
        U  = op.w_levy_flight(S)
        X  = op.replace_if_random(X, U)
        
        X = op.drop_worst(X, pr, int(k))
        
        [Xi.getFitness() for Xi in X]
    return Solution
    
def ge_190320_01(n, my_func, bounds, dimension, max_nfe):#n=100
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X]
    Solution.updateHistory(X)
    while Solution.nfe < max_nfe:
        U = X
        #Round 1
        S1 = op.select_current(U)
        U = op.w_pso(S1, w=0.75, c1=0.25, c2=1.00)
        X  = op.replace_if_random(X, U)
        #Round 2
        S1 = op.select_tournament(U, n=1, k=int(n*0.50))
        S2 = op.select_random(X, 1)
        S3 = op.select_current(X)
        U  = op.w_mut_de(S1, S2, S3, beta=0.50)
        X  = op.replace_if_best(X, U)
        #Round 3
        S1 = op.select_current(U)
        U  = op.w_pso(S1, w=0.50, c1=0.25, c2=0.00)
        X  = op.replace_if_random(X, U)
        [Xi.getFitness() for Xi in X]
    return X
    
def ge_190320_02(n, my_func, bounds, dimension, max_nfe):#n=50
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X]
    Solution.updateHistory(X)
    while Solution.nfe < max_nfe:
        U = X
        #Round 1
        S1 = op.select_tournament(U, n=1, k=1)
        U  = op.w_pso(S1, w=0.25, c1=0.00, c2=0.75)
        X  = U
        #Round Drop
        X = op.drop_worst(X, pr=0.10, k=int(n*0.25)) 
        [Xi.getFitness() for Xi in X]
    return X
    
def ge_190320_03(n, my_func, bounds, dimension, max_nfe):#n=100
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_truncate
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X]
    Solution.updateHistory(X)
    while Solution.nfe < max_nfe:
        U = X
        #Round 1
        S1 = op.select_tournament(U, n=1, k=int(n*0.25))
        S2 = op.select_random(X, 1)
        U  = op.w_crx_exp2(S1, S2, pr=0.75)
        X  = op.replace_if_random(X, U)
        #Round 2
        S1 = op.select_random(U, 1)
        S2 = op.select_random(X, 1)
        S3 = op.select_current(U)
        U  = op.w_mut_de(S1, S2, S3, beta=0.50)
        X  = U
        [Xi.getFitness() for Xi in X]
    return X
    
def ge_190325_01(n, my_func, bounds, dimension, max_nfe):#n=50
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X]
    Solution.updateHistory(X)
    while Solution.nfe < max_nfe:
        U = X
        #Round 1
        S1 = op.select_current(X)
        U  = op.w_pso(S1, w=0.50, c1=0.00, c2=2.00)
        X  = op.replace_if_random(X, U)
        #Round Drop
        X = op.drop_probability(X, pr=0.25)
        [Xi.getFitness() for Xi in X]
    return X
    
def ge_190325_03(n, my_func, bounds, dimension, max_nfe):#n=200
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    Solution.repair = op.repair_random
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X]
    Solution.updateHistory(X)
    while Solution.nfe < max_nfe:
        U = X
        #Round 1
        S1 = op.select_random(U, 1)
        U  = op.w_pso(S1, w=0.00, c1=0.00, c2=2.00)
        X  = U
        [Xi.getFitness() for Xi in X]
    return X    