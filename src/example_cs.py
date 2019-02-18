import numpy as np
from src import *

import testFunctions as tf
from animation import animation, animation3D

def cs(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    #instantiate solutions 
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        #1. Select individuals for modification in this round
        # none - select all. Alternative (bee algorythm) is to select only solutions drawn with fitness-dependant probability
        #2. de_operator = create an alternative set of solutions X1 using mutation+crossover
        X1  = op.op_pso(X, op.select_random, op.mut_cs, op.crx_exponential)
        #3. Select individual for the next generation
        X = op.replace_if_random(X, X1)
        #4 - Drop round
        X = op.drop_worst(X)

        Solution.updateHistory(X)
    return Solution, X
    
resultCS = cs(n=250, my_func=tf.katsuura, bounds=(-10, 10), dimension=20, max_nfe=250*250)
resultCS[0].print()
animation(resultCS[0].history, resultCS[0].function, *resultCS[0].bounds)