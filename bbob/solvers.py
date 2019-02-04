from __future__ import absolute_import, division, print_function
import numpy as np
from math import *
from helper import sw

# ===============================================
# the most basic example solver
# ===============================================
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


class swarmge(sw):
  def __init__(self, n, function, lb, ub, dimension, iterations):
    super(swarmge, self).__init__()
    self._agents = np.random.uniform(lb, ub, (n, dimension))
    self._points(self._agents)
    Pbest = self._agents[np.array([function(x)for x in self._agents]).argmin()]
    Gbest = Pbest
    velocity = np.zeros((n, dimension))
    nest = 100
    self._nests = np.random.uniform(lb, ub, (nest, dimension))
    for t in range(iterations):
      self._velocity(Pbest, Gbest, n, dimension, velocity)
      swap = 1
      for i in self._agents: 
        val = np.random.randint(0, nest - 1)
        if function(i) < function(self._nests[val]):
          self._nests[val] = i
      self._ordered_swap(n, nest, function)
      self._agents = np.clip(self._agents, lb, ub)
      self._nests = np.clip(self._nests, lb, ub)
      self._points(self._agents)
      if swap:
        Pbest = self._nests[np.array([function(x) for x in self._nests]).argmin()]
      else:
        Pbest = self._agents[np.array([function(x) for x in self._agents]).argmin()]
      if function(Pbest) < function(Gbest):
        Gbest = Pbest
    self._set_Gbest(Gbest)


