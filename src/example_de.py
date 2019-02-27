from src.solution import *
import src.operators as op
import testFunctions as tf
import numpy as np
from animation import animation

def pso(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X], "nfe: "+str(Solution.nfe), Solution.best.fitness
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        #FIRST OPERATOR
        #select
        S1 = op.select_random(X, 1)
        S2 = op.select_random(X, 1)
        S3 = op.select_random(X, 1)
        S  = np.append(S1, S2, axis=1)
        S  = np.append(S, S3, axis=1)
        #move
        U1 = op.w_mut_de(S)
        #update-rule
        ## keep in the pool (handled by grammar)
        
        #SECOND OPERATOR
        #select
        S1 = op.select_current(X)
        S2 = op.select_current(U1) # select from the pool (handled by grammar)
        S  = np.append(S1, S2, axis=1) 
        #move
        U2 = op.w_crx_exp(S)
        #update-rule
        X = op.replace_if_best(X, U2)
        
        Solution.updateHistory(X)
    
    return Solution, X

    
op.param = {"w":.5,"c1":1,"c2":1,"alpha":.7,"beta":.3,"pr":.6}
Solution.repair = op.repair_truncate
    
result = pso(n=100, my_func=tf.katsuura, bounds=(-10, +10), dimension=10, max_nfe=5000)
result[0].print()
animation(result[0].history, result[0].function, *result[0].bounds)