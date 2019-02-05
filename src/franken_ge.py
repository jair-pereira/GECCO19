import numpy as np
import src

import testFunctions as tf
from animation import animation, animation3D

n = 30
iteration = 30
my_func = tf.ackley_function
dimension = 10
bounds = -5, 5
beta = .5
pr = .7
tournament = 5
w = .5
c1 = .5
c2 = 1
pa = .25
dp = .1
def ge():
  X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
  [Xi.initRandom() for Xi in X]
  src.solution.updateHistory(X)
  for it in range(iteration):
    X1 = src.op.op_de(X, src.op.select_random, src.op.mut_de, src.op.crx_exponential)
    X = src.op.replace_if_random(X, X1)
    X1 = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
    X = X1
    src.solution.updateHistory(X)
  return X

ge()


animation(src.solution.history, my_func, *bounds)
#print(src.solution.best.getFitness())