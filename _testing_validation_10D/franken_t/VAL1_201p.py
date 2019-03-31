import numpy as np
import pandas as pd
import sys, pickle, datetime
import cocoex, cocopp
from solvers import f201p

output_folder = "F201p_VAL1" #CHANGE HERE!
nfe_base = 1e+5

observer = cocoex.Observer("bbob", "result_folder: " + output_folder)

suite = cocoex.Suite("bbob", "", "function_indices:2-14,16-24 dimensions:10 instance_indices:1-5")
print("Experiment ", output_folder,  " started at ",datetime.datetime.now(), "\n on suite", suite)
for problem in suite:
    problem.observe_with(observer)
    max_nfe = nfe_base*problem.dimension
    
    f201p(400, problem, (problem.lower_bounds[0], problem.upper_bounds[0]), problem.dimension, max_nfe, w_1=1.29, c1_1=0.10, c2_1=0.60, k=25, beta=0.05, w_2=0.03, c1_2=0.03, c2_2=0.52)
    
    print(problem.id, " finished at ",datetime.datetime.now())
    
cocopp.main(observer.result_folder)