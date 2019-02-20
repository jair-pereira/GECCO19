from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import testFunctions as tf



class fitness(base_ff):

    maximise = True
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        # self.iterations = params['ITERATIONS']


    def evaluate(self, ind, **kwargs):    	
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
        
        #initialize solutions 
        
        [Xi.initRandom() for Xi in X]

        for it in range(iteration):
            p, d = ind.phenotype, {}
            
            try:
                exec(p, {'X': X}, d)
            except Exception as err:
                print(p)
                print(err)
                raise err

            X = d['X']

        return src.solution.best.getFitness()


n = 30
iteration = 20

my_func   = tf.ackley_function
dimension = 10
bounds    = -5, 5

beta = .5 
pr = .7
tournamment = 5
w = .5 
c1 = .5 
c2 = 1