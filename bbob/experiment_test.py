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



# fmin = sw.pso #mod181221



def bbob(solver, output_folder):
    ### input
    suite_name = "bbob"
    output_folder = output_folder
    budget_multiplier = 100  # increase to 10, 100, ...
    solver = solver

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        # apply restarts while neither the problem is solved nor the budget is exhausted
        while (problem.evaluations < problem.dimension * budget_multiplier
               and not problem.final_target_hit):
            
            sw.pso(10, problem, problem.lower_bounds, problem.upper_bounds, problem.dimension, 3) 
            
        minimal_print(problem, final=problem.index == len(suite) - 1)

    ### post-process data
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


bbob("", "")

