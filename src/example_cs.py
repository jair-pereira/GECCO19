import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

def cs():
    #instantiate solutions 
    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    #initialize solutions 
    [Xi.initRandom() for Xi in X]
    
    #just so we can have some animations
    src.solution.updateHistory(X) # it is not necessary for the grammar
    
    for it in range(iteration):
        #1. Select individuals for modification in this round
        # none - select all. Alternative (bee algorythm) is to select only solutions drawn with fitness-dependant probability
        #2. de_operator = create an alternative set of solutions X1 using mutation+crossover
        X1  = src.op.op_pso(X, src.op.select_random,src.op.mut_cs, src.op.crx_exponential)
        #3. Select individual for the next generation
        X = src.op.replace_if_random(X, X1)
        #4 - Drop round
        X = src.op.drop_worst(X)

        src.solution.updateHistory(X) 

    return X
    
##param
n = 50
iteration = 50

my_func   = tf.ackley_function
dimension = 5
bounds    = -10, 10

beta = .5 
pr = .7
tournamment = 5
w = .5 
c1 = .5 
c2 = 1
pa = .25
dp = .1

cs()
# src.solution.best.getFitness()
animation(src.solution.history, my_func, *bounds)