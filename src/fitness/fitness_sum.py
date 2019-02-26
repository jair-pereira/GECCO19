from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import testFunctions as tf



class fitness_sum(base_ff):

    maximise = False
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.max_nfe = params['EVALUATIONS']
        self.dimension = params['DIMENSION']
        self.my_func = params['FUNCTION']
        self.bounds = params['BOUNDS']
        self.runs = params['RUNS']

    def evaluate(self, ind, **kwargs):      
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        functions = [x.strip() for x in self.my_func.split(',')]
        
        # Exec the phenotype.
        results = np.zeros((len(functions), self.runs))
        for i, f in enumerate(functions):
            d = {"max_nfe": self.max_nfe, "dimension": self.dimension, "my_func": eval(f), "bounds": self.bounds}
            for j in range(self.runs):
                try:
                    p = ind.phenotype
                    exec(p, d)
                    results[i, j] = d['XXX_output_XXX']
                except Exception as err:
                    print(p)
                    print(err)
                    raise err
        

        # Get the output
        if params['SUMMARY'] == "median":
            return np.median(results)  # this is the program's output: a number.
        else:
            return np.mean(results)

