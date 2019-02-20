import numpy as np
from math import gamma, pi, sin
from .solution import *

### INITIALIZATION METHODS ###
def init_random(lb, ub, dimension):
    return np.random.uniform(lb, ub, dimension)

### OPERATORS ###
## BLEND
# legacy
def op_blend(X, sel, mut, cross):
    sel = selection_for_op_de(X, sel)
    U = Solution.initialize(X.shape[0])
    u = np.array([mut(X[k].x, X[l].x) for k,l,m,n in sel])
    
    for i in range(len(U)): 
        U[i].setX(u[i]) 
    
    return np.array(U)
# wrapper doing it
def op_blend(S, alpha):
    u = np.array([crx_blend(S[k].x, S[l].x) for k,l in sel])
    
    for i in range(len(U)): 
        U[i].setX(u[i]) 
    
    return np.array(U)
# main
def crx_blend(x1, x2):
    #based on deap's implementation (https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py)
    gamma = (1 + 2*param['alpha']) * np.random.uniform(0, 1) - param['alpha']
    u = (1 - gamma)*x1 + gamma*x2
    v = gamma*x1 + (1 - gamma)*x2
    
    return u, v


#auxilary function for op_de
def selection_for_op_de(X, sel):
    idx_tmp = np.arange(X.shape[0])
    idx = np.array([np.append([i], sel(X, np.delete(idx_tmp, i), 3, replace=False)) for i in range(X.shape[0])])

    return idx


#TODO:
#def select_tournament(X, array, k=1, replace=True, **param): # randomly select 5(param['tournament']) Solution and return k best of them



def select_random(X, array, k, replace=False):
    return np.random.choice(array, k, replace=replace)


#TODO:
#def select_for_op(X, **params): #roulette-style selector of Solutions indicies to pass into operator functions (2nd step of abs update)


#wrapper for op_de (maybe later we can generalize a wrapper for all operators)
#operators work at Solutions, a np.array
#wrapper work at an object

#for now, lets make separate functions for each operator but we'll introduce some variation with parameters
#all "op_" functions produce an alternative population of Solutions X1 which we will accept or regect at output stage

def op_de(X, sel, mut, cross):
    sel = selection_for_op_de(X, sel)
    # U = Solution.initialize(X.shape[0])
    u = np.array([apply_op_de(X[k], X[l], X[m], X[n], mut, cross) for k,l,m,n in sel])
    
    for i in range(len(X)): 
        X[i].setX(u[i]) 
    
    return X#np.array(U)

#PSO-operator. Updates each Solution that is passed to it with one of the <mut> step operators
def op_pso(X, sel, mut, cross): # this function will recieve some type of select and crossover parameters but will not use them
    sel = selection_for_op_de(X, sel)
    # U = Solution.initialize(X.shape[0])
    u = np.array([mut(X[k], X[l], X[m]) for k, l, m, n in sel])
    
    for i in range(len(X)): 
        X[i].setX(u[i])

    return X#np.array(U)




    
def op_mutU(X, sel, mut, cross):
    sel = selection_for_op_de(X, sel)
    U = Solution.initialize(X.shape[0])
    u = np.array([mut(X[k].x, *X[k].bounds) for k,l,m,n in sel])
    
    for i in range(len(U)): 
        U[i].setX(u[i]) 
    
    return np.array(U)
    


#auxilary function to op_de
def apply_op_de(xi, xr1, xr2, xr3, mut, cross):
    u = mut(xr1, xr2, xr3)
    v, _ = cross(xi.x, u)
    return v

def mut_uniform(x, lb, ub):
    u = [np.random.uniform(lb, ub) if np.random.random() < param['pr'] else x[i] for i in range(len(x))]
    
    return u

def mut_de(x1, x2, x3):
    u = x1.x + param['beta']*(x2.x-x3.x)
    return u

#pso velocity update (potentially can be called from op_de on <mut> operator)
def mut_pso(x1, x2, x3):
    r1 = np.random.random(x1.x.shape)
    r2 = np.random.random(x1.x.shape)
    x1.velocity = param['w']*x1.velocity + param['c1']*r1*(x1.pbest['x'] - x1.x) + param['c2']*r2*(Solution.best.x - x1.x)
    u = x1.velocity + x1.x
        
    return u

#cs Levi flight
def mut_cs(x1, x2, x3): 
    beta = 3 / 2
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    w = np.array(np.random.standard_normal(x1.x.shape)) * sigma
    v = np.array(np.random.standard_normal(x1.x.shape))
    step = w / abs(v) ** (1 / beta)

    x1.getFitness() 
    stepsize = 0.2 * step * (x1.x - x1.pbest['x'])
    u = x1.x + stepsize

    return u

def crx_npoint(x1, x2, points):
    u = np.array([_ for _ in x1])
    v = np.array([_ for _ in x2])
    
    u[points] = v[points]
    v[points] = u[points]
    
    return u, v

def crx_exponential(x1, x2, func=crx_npoint):
    all_points = np.arange(x1.shape[0])
    i = np.random.choice(all_points)    #ensure at least one point
    crossover_points = [all_points[i]]

    while param['pr'] >= np.random.uniform(0, 1) and len(crossover_points) < len(all_points):
        i = (i+1) % len(all_points)
        crossover_points = crossover_points + [all_points[i]]
        
    
    u, v = func(x1, x2, crossover_points)
    return u, v

    
def replace_if_best(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[i].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)

#cuckoo-style update
def replace_if_random(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[np.random.randint(0, X1.shape[0])].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)
    return X2

#TODO:
def drop_probability(X):
    for i in range(X.shape[0]):
        if np.random.random() < param['pr']:
            X[i].setX(init_random(*type(X[i]).bounds, type(X[i]).dimension))
            X[i].getFitness()
    return X


#TODO:
def drop_worst(X):
    u = np.array([(X[i].getFitness(), i) for i in range(X.shape[0])])
    u = sorted(u, key=lambda x:x[0])
    for i in range(int(len(X)*.25)):
        if np.random.random() < param['pr']:
            ind = int(u[i][1])
            X[ind].setX(init_random(*type(X[ind]).bounds, type(X[ind]).dimension))
    return X

#TODO:
#def drop_old():
