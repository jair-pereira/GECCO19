import numpy as np
from math import gamma, pi, sin
from . import solution

beta = .5 
pr = .7 
tournamment = 5
w = .5 
c1 = .5 
c2 = 1
pa = .25
dp = .1
blend_alpha = .5
pr_uni = .5

#auxilary function for op_de
def selection_for_op_de(X, sel):
    idx_tmp = np.arange(X.shape[0])
    idx = np.array([np.append([i], sel(X, np.delete(idx_tmp, i), 3, replace=False)) for i in range(X.shape[0])])

    return idx


#TODO:
#def select_tournament(X, array, k=1, replace=True, **param): # randomly select 5(param['tournament']) solution and return k best of them



def select_random(X, array, k, replace=False):
    return np.random.choice(array, k, replace=replace)


#TODO:
#def select_for_op(X, **params): #roulette-style selector of solutions indicies to pass into operator functions (2nd step of abs update)


#wrapper for op_de (maybe later we can generalize a wrapper for all operators)
#operators work at solutions, a np.array
#wrapper work at an object

#for now, lets make separate functions for each operator but we'll introduce some variation with parameters
#all "op_" functions produce an alternative population of solutions X1 which we will accept or regect at output stage

def op_de(X, sel, mut, cross): 
    sel = selection_for_op_de(X, sel)
    U = np.array([solution(X[0].function, X[0].x.shape[0], X[0].limits) for i in range(X.shape[0])])
    u = np.array([apply_op_de(X[k], X[l], X[m], X[n], mut, cross) for k,l,m,n in sel])
    
    for i in range(len(U)): 
        U[i].setX(u[i]) 
    
    return np.array(U)

#PSO-operator. Updates each solution that is passed to it with one of the <mut> step operators
def op_pso(X, sel, mut, cross): # this function will recieve some type of select and crossover parameters but will not use them
    sel = selection_for_op_de(X, sel)
    U = np.array([solution(X[0].function, X[0].x.shape[0], X[0].limits) for i in range(X.shape[0])])
    u = np.array([mut(X[k], X[l], X[m]) for k, l, m, n in sel])
    
    for i in range(len(U)): 
        U[i].setX(u[i]) 

    return np.array(U)


#TODO:
#def op_ga(X, sel, mut, cross, **param): 


#auxilary function to op_de
def apply_op_de(xi, xr1, xr2, xr3, mut, cross):
    u = mut(xr1, xr2, xr3)
    v, _ = cross(xi.x, u)
    return v

def mut_uniform(x, lb, ub):
    u = [np.random.uniform(lb, ub) if np.random.random() < pr_uni else x[i] for i in range(len(x))]
    
    return u

def mut_de(x1, x2, x3):
    u = x1.x + beta*(x2.x-x3.x)
    return u

#pso velocity update (potentially can be called from op_de on <mut> operator)
def mut_pso(x1, x2, x3): 
    r1 = np.random.random(x1.x.shape)
    r2 = np.random.random(x1.x.shape)
    x1.getFitness()
    x1.velocity = w*x1.velocity + c1*r1*(x1.pbest_x - x1.x) + c2*r2*(solution.best.x - x1.x)
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
    stepsize = 0.2 * step * (x1.x - x1.pbest_x)
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

    while pr >= np.random.uniform(0, 1) and len(crossover_points) < len(all_points):
        i = (i+1) % len(all_points)
        crossover_points = crossover_points + [all_points[i]]
        
    
    u, v = func(x1, x2, crossover_points)
    return u, v

def crx_blend(x1, x2):
    #based on deap's implementation (https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py)
    gamma = (1 + 2*blend_alpha) * np.random.uniform(0, 1) - blend_alpha
    u = (1 - gamma)*x1 + gamma*x2
    v = gamma*x1 + (1 - gamma)*x2
    
    #return u
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
        if np.random.random() < dp:
            X[i].initRandom()
            X[i].getFitness()
    return X


#TODO:
def drop_worst(X):
    [X[i].getFitness() for i in range(X.shape[0])]
    u = np.array([(X[i].fitness, i) for i in range(X.shape[0])])
    u = sorted(u, key=lambda x:x[0])
    for i in range(20):
        if np.random.random() < pa:
            ind = int(u[i][1])
            X[ind].initRandom()
    return X

#TODO:
#def drop_old():
