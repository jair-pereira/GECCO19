#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence it neither produces any progress messages (which can be very
annoying for long experiments) nor provides batch distribution,
as `example_experiment.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

    import cma
    def fmin(fun, x0):
        return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser

from solvers import *

### input
suite_name = "bbob"
suite_dim = "dimensions:2"
output_folder = "comp"
budget_multiplier = 20000 # increase to 10, 100, ...

#general param
n = 100
iteration = 20

#pso: w, c1, c2
params_pso = (1,1,1)

#de: beta, crossover prob
params_de = (.7, .8) 

#sa: T
T = sa.temperature_exp(1000, .2, 100)
# T = sa.temperature_lin(1000, -5, 100)

### prepare
suite = cocoex.Suite(suite_name, "", suite_dim) #Suite("bbob", "year:2009", "dimensions:20 instance_indices:1"
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while (problem.evaluations < problem.dimension * budget_multiplier and not problem.final_target_hit):
        
        pso(n, iteration, problem.dimension, problem, problem.lower_bounds, problem.upper_bounds, *params_pso)
#         sa(n, problem.dimension, problem, problem.lower_bounds, problem.upper_bounds, T)
#         de(n, iteration, problem.dimension, problem, problem.lower_bounds, problem.upper_bounds, beta, *params_de)
        
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

# for suite_name in cocoex.known_suite_names:
#     suite = cocoex.Suite(suite_name, "", "")
#     print(suite.dimensions)