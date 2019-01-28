import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

def de():
    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    [Xi.initRandom() for Xi in X]
    src.solution.updateHistory(X)
    
    
    for it in range(iteration):
        sel = src.op.selection_for_op_de(X.shape[0])
        X1  = src.op.apply_op_de(src.op.op_de, X, sel, (crx, beta, pr))

        X = src.op.selection_de(X, X1)
        
        src.solution.updateHistory(X) 
    
    r = np.argmax([Xi.getFitness() for Xi in X])
    
    return X[r]

##param
n = 50
iteration = 100

my_func   = tf.ackley_function
dimension = 5
bounds    = -10, 10

crx  = src.op.crx_exponential
beta = .5
pr   = .7

r = de()
# src.solution.best.getFitness()
animation(src.solution.history, my_func, *bounds)