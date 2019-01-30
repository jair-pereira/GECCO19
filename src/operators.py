import numpy as np
from . import solution

# n: pop_size
def selection_for_op_de(X, sel, **param):
    idx_tmp = np.arange(X.shape[0])
    idx = np.array([np.append([i], sel(X, np.delete(idx_tmp, i), 3, replace=False), **param) for i in range(X.shape[0])])

    return idx


# TODO:
# def select_tournament(X, array, k=1, replace=True, **param): # randomly select 5(param['tournament']) solution and return k best of them



def select_random(X, array, k=1, replace=True, **param):
    return np.random.choice(array, k, replace=replace)


# TODO:
# def select_for_op(X, **params): #roulette-style selector of solutions indicies to pass into operator functions


#wrapper for op_de (maybe later we can generalize a wrapper for all operators)
#operators work at solutions, a np.array
#wrapper work at an object

#for simplicity, lets make separate functions for each operator but we'll introduce some variation with parameters
#all "op_" functions produce an alternative population of solutions X1 which we will accept or regect at output stage

def apply_op_de(X, sel, func, **param):
    sel = selection_for_op_de(X, sel, **param)
    U = np.array([solution(X[0].function, X[0].x.shape[0], X[0].limits) for i in range(X.shape[0])])
    u = np.array([op_de(X[k].x, X[l].x, X[m].x, X[n].x, func, **param) for k,l,m,n in sel])
    
    for i in range(len(U)): #
        U[i].setX(u[i]) 
    
    return np.array(U)

#TODO:
# def op_pso(X, sel, func, **param): 


#TODO:
# def op_ga(X, sel, func, **param): 



def op_de(xi, xr1, xr2, xr3, func, **param):
    u = mut_de(xr1, xr2, xr3, param['beta'])
    v, _ = func(xi, u, param['pr'])
    return v

def mut_de(x1, x2, x3, beta):
    u = x1 + beta*(x2-x3)
    return u

def crx_npoint(x1, x2, points):
    u = np.array([_ for _ in x1])
    v = np.array([_ for _ in x2])
    
    u[points] = v[points]
    v[points] = u[points]
    
    return u, v

def crx_exponential(x1, x2, pr, func=crx_npoint):
    all_points = np.arange(x1.shape[0])
    i = np.random.choice(all_points)    #ensure at least one point
    crossover_points = [all_points[i]]

    while pr >= np.random.uniform(0, 1) and len(crossover_points) < len(all_points):
        i = (i+1) % len(all_points)
        crossover_points = crossover_points + [all_points[i]]
        
    
    u, v = func(x1, x2, crossover_points)
    return u, v

def replace_if_best(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[i].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)

#cuckoo-style update
def replace_if_random(X1, X2):
    U = [X2[i] if X2[i].getFitness() > np.random.choice(X1, 1).getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)
