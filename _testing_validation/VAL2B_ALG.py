import numpy as np
import pandas as pd
import sys, pickle, datetime
import cocoex, cocopp
from solvers import pso, de, cs, ga

output_folder = "CS_VAL2A_190327" #CHANGE HERE!
nfe_base = 1e+4

observer = cocoex.Observer("bbob", "result_folder: " + output_folder)

suite = cocoex.Suite("bbob", "", "function_indices:16-29 dimensions:20 instance_indices:1-5")
for problem in suite:
    problem.observe_with(observer)
    max_nfe = nfe_base*problem.dimension
    
    #CHANGE HERE!
    cs(50,problem, (problem.lower_bounds[0], problem.upper_bounds[0]), problem.dimension, max_nfe, pr=0.97, k=25)
    
    print(problem.id, " finished at ",datetime.datetime.now())
    
cocopp.main(observer.result_folder)