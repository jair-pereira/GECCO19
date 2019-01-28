import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

def de():
    #instantiate solutions 
    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    #initialize solutions 
    [Xi.initRandom() for Xi in X]
    
    #just so we can have some animations
    src.solution.updateHistory(X) # it is not necessary for the grammar
    
    for it in range(iteration):
        #1. Select individuals for de_operator
        sel = src.op.selection_for_op_de(X.shape[0]) #return indices
        #2. de_operator = mutation+crossover
        X1  = src.op.apply_op_de(src.op.op_de, X, sel, (crx, beta, pr))
        #3. Select individual for the next generation
        X = src.op.selection_de(X, X1)
        
        src.solution.updateHistory(X) 

    return X
    
##param
n = 50
iteration = 100

my_func   = tf.ackley_function
dimension = 5
bounds    = -10, 10

crx  = src.op.crx_exponential
beta = .5
pr   = .7

de()
# src.solution.best.getFitness()
animation(src.solution.history, my_func, *bounds)