from __future__ import absolute_import, division, print_function
import numpy as np
from math import *
import src

import testFunctions as tf

# ===============================================
# the most basic example solver
# ===============================================

def ge(n, my_func, lb, ub, dimension, iteration):
  X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
  [Xi.initRandom() for Xi in X]
  src.solution.updateHistory(X)
  for it in range(iteration):
    X1 = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
    X = src.op.replace_if_best(X, X1)
    X = src.op.drop_probability(X)
    src.solution.updateHistory(X)
  return X


def pso(n, my_func, lb, ub, dimension, iteration):
    #instantiate solutions 
    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    #initialize solutions 
    [Xi.initRandom() for Xi in X]
    for it in range(iteration):
        #1. Select individuals for modification in this round
        # none - select all. Alternative (bee algorythm) is to select only solutions drawn with fitness-dependant probability
        #2. de_operator = create an alternative set of solutions X1 using mutation+crossover
        X1  = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
        #3. Select individual for the next generation <- accept all
        X = X1
    return X

def de(n, my_func, lb, ub, dimension, iteration):
    #instantiate solutions 
    X = np.array([src.solution(my_func, dimension, (lb, ub)) for i in range(n)])
    #initialize solutions 
    [Xi.initRandom() for Xi in X]      
    for it in range(iteration):
        #1. Select individuals for modification in this round
        # none - select all. Alternative (bee algorythm) is to select only solutions drawn with fitness-dependant probability
        #2. de_operator = create an alternative set of solutions X1 using mutation+crossover
        X1  = src.op.op_de(X, src.op.select_random, src.op.mut_de, src.op.crx_exponential)
        #3. Select individual for the next generation
        X = src.op.replace_if_best(X, X1)
    return X

def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(max([1, min([budget, max_chunk_size])]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        if fun.number_of_constraints > 0:
            C = [fun.constraint(x) for x in X]  # call constraints
            F = [fun(x) for i, x in enumerate(X) if np.all(C[i] <= 0)]
        else:
            F = [fun(x) for x in X]
        if fun.number_of_objectives == 1:
            index = np.argmin(F) if len(F) else None
            if index is not None and (f_min is None or F[index] < f_min):
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min


# class swarmge(sw):
#   def __init__(self, n, function, lb, ub, dimension, iterations):
#     super(swarmge, self).__init__()
#     self._agents = np.random.uniform(lb, ub, (n, dimension))
#     self._points(self._agents)
#     Pbest = self._agents[np.array([function(x)for x in self._agents]).argmin()]
#     Gbest = Pbest
#     velocity = np.zeros((n, dimension))
#     nest = 100
#     self._nests = np.random.uniform(lb, ub, (nest, dimension))
#     for t in range(iterations):
#       self._velocity(Pbest, Gbest, n, dimension, velocity)
#       swap = 1
#       for i in self._agents: 
#         val = np.random.randint(0, nest - 1)
#         if function(i) < function(self._nests[val]):
#           self._nests[val] = i
#       self._ordered_swap(n, nest, function)
#       self._agents = np.clip(self._agents, lb, ub)
#       self._nests = np.clip(self._nests, lb, ub)
#       self._points(self._agents)
#       if swap:
#         Pbest = self._nests[np.array([function(x) for x in self._nests]).argmin()]
#       else:
#         Pbest = self._agents[np.array([function(x) for x in self._agents]).argmin()]
#       if function(Pbest) < function(Gbest):
#         Gbest = Pbest
#     self._set_Gbest(Gbest)


