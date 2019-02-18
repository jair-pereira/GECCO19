import numpy as np
from src import *

import testFunctions as tf
from animation import animation, animation3D

def pso(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    #instantiate solutions 
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        print(Solution.nfe)
        [Xi.getFitness() for Xi in X]
        # [Xi.updatePBest() for Xi in X]
        X1  = op.op_pso(X, op.select_random, op.mut_pso, op.crx_exponential)
        X = X1
        
        Solution.updateHistory(X)
    return Solution, X
    
resultPSO = pso(n=5, my_func=tf.katsuura, bounds=(-10, 10), dimension=20, max_nfe=5*5)
resultPSO[0].print()
animation(resultPSO[0].history, resultPSO[0].function, *resultPSO[0].bounds)