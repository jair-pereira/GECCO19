import numpy as np
from src import *

import testFunctions as tf
from animation import animation, animation3D

def de(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    #instantiate solutions 
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        #1. DE mutation -> Exponential crossover
        X1  = op.op_de(X, op.select_random, op.mut_de, op.crx_exponential)        
        #3. update rule
        X = op.replace_if_best(X, X1)
        
        Solution.updateHistory(X) 

    return Solution, X
    
resultDE = de(n=250, my_func=tf.katsuura, bounds=(-10, 10), dimension=20, max_nfe=250*250)
resultDE[0].print()
animation(resultDE[0].history, resultDE[0].function, *resultDE[0].bounds)