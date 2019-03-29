import numpy as np
import pandas as pd
import sys, pickle, datetime
import cocoex, cocopp
from solvers import pso, de, cs, ga

output_folder = "DE_VAL1" #CHANGE HERE!
nfe_base = 1e+5

observer = cocoex.Observer("bbob", "result_folder: " + output_folder)

suite = cocoex.Suite("bbob", "", "function_indices:2-14,16-24 dimensions:10 instance_indices:1-5")
print("Experiment ", output_folder,  " started at ",datetime.datetime.now(), "\n on suite", suite)
for problem in suite:
    problem.observe_with(observer)
    max_nfe = nfe_base*problem.dimension
    
    de(1600, problem, (problem.lower_bounds[0], problem.upper_bounds[0]), problem.dimension, max_nfe, beta=1.01, pr=0.93)
    
    print(problem.id, " finished at ",datetime.datetime.now())
    
cocopp.main(observer.result_folder)