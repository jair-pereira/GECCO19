from src.solution import *
import src.operators as op
import testFunctions as tf
import numpy as np
from animation import animation

def pso(n, my_func, bounds, dimension, max_nfe):
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    X = Solution.initialize(n)
    for Xi in X:    Xi.setX(op.init_random(*Solution.bounds, Solution.dimension))
    [Xi.getFitness() for Xi in X], "nfe: "+str(Solution.nfe), Solution.best.fitness
    Solution.updateHistory(X)
    
    while Solution.nfe < max_nfe:
        #select
        S = op.select_current(X)
        #move
        U = op.w_pso(S)
        #update pso pbest, gbest
        [Ui.getFitness() for Ui in U]
        #update-rule
        X = U
        
        Solution.updateHistory(X)

    
    return Solution, X

    
op.param = {"w":.5,"c1":1,"c2":1,"alpha":.7,"beta":.5,"pr":.6}
Solution.repair = op.repair_truncate
    
resultPSO = pso(n=100, my_func=tf.katsuura, bounds=(-10, +10), dimension=10, max_nfe=5000)
resultPSO[0].print()
animation(resultPSO[0].history, resultPSO[0].function, *resultPSO[0].bounds)