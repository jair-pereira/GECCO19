from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import testFunctions as tf



class fitness(base_ff):

    maximise = False
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.max_nfe = params['EVALUATIONS']
        self.dimension = params['DIMENSION']
        self.my_func = eval(params['FUNCTION'])
        self.bounds = params['BOUNDS']
        self.runs = params['RUNS']

    def evaluate(self, ind, **kwargs):    	
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        par_dict = {"max_nfe": self.max_nfe, "dimension": self.dimension, "my_func": self.my_func, "bounds": self.bounds}

        p, d = ind.phenotype, {}
        
        # Exec the phenotype.
        try:
            results = np.zeros(self.runs)
            for i in range(self.runs):
                exec(p, par_dict, d)
                results[i] = d['XXX_output_XXX']
        except Exception as err:
            print(p)
            print(err)
            raise err

        # Get the output
        if params['SUMMARY'] == "median":
            return np.median(results)  # this is the program's output: a number.
        else:
            return np.mean(results)

