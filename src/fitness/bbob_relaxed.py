from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import cocoex, cocopp  # bbob experimentation and post-processing modules
import pickle

class bbob_relaxed(base_ff):
    maximise = True
    params['M'] = 1
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        
        #parameters from ge.txt
        self.max_nfe = params['MAX_NFE']
        self.runs    = params['RUNS']
        self.suite   = cocoex.Suite(
                            "bbob", "", 
                            "function_indices:"  +str(params['FUNCTION'])+
                            " dimensions:"       +str(params['DIMENSIONS'])+ 
                            " instance_indices:" +str(params['INSTANCE_INDICES'])
                            )
        
        print("Using BBOB suite: ",self.suite)
        
        #relaxation
        file = open("bbob_final_target_fvalue1.pkl",'rb')
        self.ftarget_values = pickle.load(file)
        file.close()
        self.multiplier = params['MULTIPLIER']

    def evaluate(self, ind, **kwargs):
        d_target_hit = {} #{problem.id, nbr of target hit}
        d_fitness = {}
        
        for problem in self.suite:
            #inputs for the generated algorithm
            d = {
                "max_nfe"  : self.max_nfe, 
                "dimension": problem.dimension,
                "my_func"  : problem,
                "bounds"   : (problem.lower_bounds[0], problem.upper_bounds[0])
                }
            
            d_target_hit[problem.id] = 0
            tmp_fitness = []
            
            for j in range(self.runs):
                try:
                    p = ind.phenotype
                    exec(p, d)
                    
                    tmp_fitness.append(d['XXX_output_XXX'])
                    if (d['XXX_output_XXX'] - self.ftarget_values[problem.id]*params['M']) < 0:
                        d_target_hit[problem.id] += 1

                except Exception as err:
                    print(p)
                    print(err)
                    raise err
                    
            d_fitness[problem.id] = tmp_fitness

            
        #TODO: compute mean, median, mode, var
        # np.median()
        # np.mean()
        # np.var()
        
        result = sum(d_target_hit.values()) / len(self.suite)
        
        return result