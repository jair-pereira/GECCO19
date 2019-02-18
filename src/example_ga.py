import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

def ga():
    #instantiate solutions 
    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    #initialize solutions 
    [Xi.initRandom() for Xi in X]
    
    #just so we can have some animations
    src.solution.updateHistory(X)
    
    for it in range(iteration):
        X1  = src.op.op_blend(X, src.op.select_random, src.op.crx_blend, None)
        X = X1
        X1  = src.op.op_mutU(X, src.op.select_random, src.op.mut_uniform, None)
        X = src.op.replace_if_best(X, X1)
        
        src.solution.updateHistory(X) 


    return X
    
##param
n = 200
iteration = 50

my_func   = tf.katsuura
dimension = 20
bounds    = -10, 10

beta = .5 
pr = .7
tournamment = 5
w = .5 
c1 = .5 
c2 = 1

ga()
print(src.solution.best.getFitness())
animation(src.solution.history, my_func, *bounds)