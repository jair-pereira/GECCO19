from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
from math import *
import src
import cocoex, cocopp  # bbob experimentation and post-processing modules
from stats.stats import stats, get_stats # for our convenience 

class bbob_fitness(base_ff):
    maximise = False
    
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        
        #parameters from ge.txt
        self.max_nfe = params['MAX_NFE']
        self.n = params['SWARM_SIZE']
        self.suite   = cocoex.Suite(
                            "bbob", "", 
                            "function_indices:"  +str(params['FUNCTION'])+
                            " dimensions:"       +str(params['DIMENSION'])+ 
                            " instance_indices:" +str(params['INSTANCE_INDICES'])
                            )
        
        print("Using BBOB suite: ",self.suite)
        
        #learning method
        self._ind = -1
        self._gen = 0
        
        #log
        self.logh = open(params['FILE_PATH']+"/history.csv", 'w') #190312: log
        output_list = []
        output_list.append("gen")
        output_list.append("indv")
        output_list.append("hh_fit")
        for p in self.suite:
            output_list.append(p.id)
        self.logh.write(",".join(map(str,output_list))+"\n")
        
    # def __del__(self):
        # self.logh.close()

    def evaluate(self, ind, **kwargs):
        d_fitness = {}
        
        for problem in self.suite:
            #inputs for the generated algorithm
            d = {
                "max_nfe"  : self.max_nfe,
                "n": self.n, 
                "dimension": problem.dimension,
                "my_func"  : problem,
                "bounds"   : (problem.lower_bounds[0], problem.upper_bounds[0])
                }
            
            try:
                p = ind.phenotype
                exec(p, d)
                    
                tmp_fitness = d['XXX_output_XXX']
                        
            except Exception as err:
                print(p)
                print(err)
                raise err
                    
            d_fitness[problem.id] = tmp_fitness
      
        result = sum(d_fitness.values())
        
        ### log ###
        self._ind += 1
        if stats['gen'] > self._gen:
            self._gen = stats['gen']
            self._ind = 0
        output_list = []
        output_list.append(stats['gen'])
        output_list.append(self._ind)
        output_list.append(result)
        self.logh.write(",".join(map(str,output_list))+"\n")
        self.logh.flush()
        ###
        
        return result