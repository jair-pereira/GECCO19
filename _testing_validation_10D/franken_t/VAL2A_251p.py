import numpy as np
import pandas as pd
import sys, pickle, datetime
import cocoex, cocopp
from solvers import f251p

output_folder = "F251p_VAL2A" #CHANGE HERE!
nfe_base = 1e+5

observer = cocoex.Observer("bbob", "result_folder: " + output_folder)

suite = cocoex.Suite("bbob", "", "function_indices:2-5 dimensions:10 instance_indices:1-5")
print("Experiment ", output_folder,  " started at ",datetime.datetime.now(), "\n on suite", suite)
for problem in suite:
    problem.observe_with(observer)
    max_nfe = nfe_base*problem.dimension
    
    f251p(100, problem, (problem.lower_bounds[0], problem.upper_bounds[0]), problem.dimension, max_nfe, w=0.06, c1=0.00, c2=1.44, pr=0.08)
    
    print(problem.id, " finished at ",datetime.datetime.now())
    
cocopp.main(observer.result_folder)