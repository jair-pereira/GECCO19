import numpy as np
import pandas as pd
import sys, pickle, datetime
import cocoex, cocopp
from solvers import f201p, f203p, f251p, f253p

output_folder = "F253p_TEST_190327" #CHANGE HERE!
nfe_base = 1e+5

observer = cocoex.Observer("bbob", "result_folder: " + output_folder)

suite = cocoex.Suite("bbob", "", "function_indices:1,15 dimensions:10,20 instance_indices:1-10")
print("Experiment ", output_folder,  " started at ",datetime.datetime.now(), "\n on suite", suite)
for problem in suite:
    problem.observe_with(observer)
    max_nfe = nfe_base*problem.dimension
    
    f253p(100, problem, (problem.lower_bounds[0], problem.upper_bounds[0]), problem.dimension, max_nfe, w=0.61, c1=0.00, c2=0.70)
    
    print(problem.id, " finished at ",datetime.datetime.now())
    
cocopp.main(observer.result_folder)