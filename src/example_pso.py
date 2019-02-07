import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

def pso():
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
        X1  = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
        #3. Select individual for the next generation <- accept all
        X = X1
        
        src.solution.updateHistory(X) 


    return X
    
##param
n = 30
iteration = 20

my_func   = tf.katsuura
dimension = 40
bounds    = -5, 5

beta = .5 
pr = .7
tournamment = 5
w = .5 
c1 = .5 
c2 = 1

pso()
#print(src.solution.best.getFitness())

animation(src.solution.history, my_func, *bounds)