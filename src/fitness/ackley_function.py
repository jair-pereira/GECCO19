from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
from animation import animation, animation3D



class ackley_function(base_ff):

    maximise = True
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):    	
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        p, d = ind.phenotype, {}
        
        # Exec the phenotype.
        try:
            exec(p, d)
        except Exception as err:
            print(p)
            print(err)
            raise err

        # Get the output
        s = d['XXX_output_XXX']  # this is the program's output: a number.
        
        
        return s