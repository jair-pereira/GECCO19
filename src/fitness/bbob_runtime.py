from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import cocoex, cocopp  # bbob experimentation and post-processing modules


class bbob_runtime(base_ff):

    maximise = False
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.max_nfe = params['MAX_NFE']
        self.runs = params['RUNS']
        self.suite = cocoex.Suite("bbob", "", "function_indices:"+ str(params['FUNCTION'])+ "dimensions:"+ str(params['DIMENSIONS']) + 
            "instance_indices:" + str(params['INSTANCE_INDICES']))

    def evaluate(self, ind, **kwargs):      
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        
        runtime = []
        for i, problem in enumerate(self.suite):
            d = {"max_nfe": self.max_nfe, "dimension": problem.dimension, "my_func": problem, "bounds": (problem.lower_bounds, problem.upper_bounds)}
            results = np.zeros(self.runs)
            while (problem.evaluations < self.runs * self.max_nfe and not problem.final_target_hit):
                try:
                    # Exec the phenotype.
                    p = ind.phenotype
                    exec(p, d)
                except Exception as err:
                    print(p)
                    print(err)
                    raise err
            runtime.append(problem.evaluations)

        # Get the output
        if params['SUMMARY'] == "median":
            return np.median(runtime)
        elif params['SUMMARY'] == "mean":
            return np.mean(runtime)
        elif params['SUMMARY'] == "var":
            return np.var(runtime)
        else:
            return np.min(runtime)

