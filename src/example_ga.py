import numpy as np
from src import *

import testFunctions as tf
from animation import animation, animation3D

def ga(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    #instantiate solutions 
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        #1. blend crossover
        X1  = op.op_blend(X, op.select_random, op.crx_blend, None)
        #1. update-rule
        # X = X1
        #2. uniform mutation
        X1  = op.op_mutU(X, op.select_random, op.mut_uniform, None)
        #2. update-rule 
        X = op.replace_if_best(X, X1)
        
        Solution.updateHistory(X)
    return Solution, X

resultGA = ga(n=250, my_func=tf.katsuura, bounds=(-10, 10), dimension=20, max_nfe=250*250)
resultGA[0].print()
animation(resultGA[0].history, resultGA[0].function, *resultGA[0].bounds)