import numpy as np
from . import solution

# n: pop_size
def selection_for_op_de(n):
    idx_tmp = np.arange(n)
    idx = np.array([np.append([i], np.random.choice(np.delete(idx_tmp, i), 3, replace=False)) for i in range(n)])

    return idx

def selection_de(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[i].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)

#wrapper for op_de (maybe later we can generalize a wrapper for all operators)
#operators work at solutions, a np.array
#wrapper   work at an object
def apply_op_de(X, sel, func, **param):
    sel = sel(X.shape[0])
    U = np.array([solution(X[0].function, X[0].x.shape[0], X[0].limits) for i in range(X.shape[0])])
    u = np.array([op_de(X[k].x, X[l].x, X[m].x, X[n].x, func, **param) for k,l,m,n in sel])
    
    for i in range(len(U)):
        U[i].setX(u[i])
    
    return np.array(U)

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