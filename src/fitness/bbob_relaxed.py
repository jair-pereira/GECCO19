from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import cocoex, cocopp  # bbob experimentation and post-processing modules


class bbob_relaxed(base_ff):

    maximise = False
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.max_nfe = params['MAX_NFE']
        self.runs    = params['RUNS']
        self.suite   = cocoex.Suite("bbob", "", "function_indices:"+ str(params['FUNCTION'])+ " dimensions:"+ str(params['DIMENSIONS']) + 
            " instance_indices:" + str(params['INSTANCE_INDICES']))
            
        self.multiplier    = params['MULTIPLIER']

    def evaluate(self, ind, **kwargs):      
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        
        success = []
        fitness = []
        for problem in self.suite:
            d = {"max_nfe": self.max_nfe, "dimension": problem.dimension, "my_func": problem, "bounds": (problem.lower_bounds, problem.upper_bounds)}
            
            try:
                # Exec the phenotype.
                p = ind.phenotype
                exec(p, d)
                fitness.append(d['XXX_output_XXX'])
                ###
                # success.append(problem.final_target_hit)
                #if d['XXX_output_XXX'] (problem.final_target_fvalue1 * self.multiplier):
                                       
                    
                    ###
            except Exception as err:
                print(p)
                print(err)
                raise err

        # Get the output
        if params['SUMMARY'] == "median":
            return np.median(fitness)
        elif params['SUMMARY'] == 'mean':
            return np.mean(fitness)
        elif params['SUMMARY'] == "var":
            return np.var(fitness)
        elif params['SUMMARY'] == "success":
            maximise = True
            return sum(success)
        else:
            return np.min(fitness)

