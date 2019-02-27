import numpy as np
from math import gamma, pi, sin
from .solution import *

### INITIALIZATION METHODS ###
def init_random(lb, ub, dimension):
    return np.random.uniform(lb, ub, dimension)
    
### SELECTION METHODS (INPUT) ###
def select_random(X, n=1):
    """ Selects n exclusively and randomly candidate solutions from X
    :param X: list of candidate solutions
    :param n: the number of individuals selected
    :returns: A list of references of the selected candidate solutions. """
    s = [np.random.choice(X, n, replace=False) for _ in range(len(X))]
    return np.array(s)
    
def select_tournament(X, n, k): 
    """ Selects n exclusively candidate solutions from X through tournament of group size k
    :param X: list of candidate solutions
    :param n: the number of individuals selected
    :param k: group size
    :returns: A list of references of the selected candidate solutions. """
    s = [np.sort(r)[::-1][:n] for r in select_random(X, k)]
    return np.array(s)

def select_current(X):
    """ Selects all candidate solutions from X
    :param X: list of candidate solutions
    :returns: A list of references of the selected candidate solutions. """
    s = [[Xi] for Xi in X]
    return np.array(s)

### OPERATORS ###  
## CROSSOVER BLEND
# wrapper_2children
def w_crx_blend(S1, S2, alpha):
    U = Solution.initialize(2*len(S1))
    u = np.array([crx_blend(X1.x, X2.x, alpha) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i, Ui in enumerate(U[0::2]):
        Ui.setX(u[i,0])
        # Ui.copyStatusPSO(S1[i,0])
        Ui.setVelocity(S1[i,0].velocity)
        Ui.pbest = S1[i,0].pbest
        
    for i, Ui in enumerate(U[1::2]):
        Ui.setX(u[i,1])
        # Ui.copyStatusPSO(S2[i,0])
        Ui.setVelocity(S2[i,0].velocity)
        Ui.pbest = S2[i,0].pbest
    
    return U
    
# wrapper_1child
def w_crx_blend2(S1, S2, alpha):
    U = Solution.initialize(len(S1))
    u = np.array([crx_blend(X1.x, X2.x, alpha) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return U
    
# main
def crx_blend(x1, x2, alpha):
    #based on deap's implementation (https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py)
    gamma = (1 + 2*alpha) * np.random.uniform(0, 1) - alpha
    u = (1 - gamma)*x1 + gamma*x2
    v = gamma*x1 + (1 - gamma)*x2
    
    return u, v
    
## CROSSOVER EXPONENTIAL
# wrapper 2children
def w_crx_exp(S1, S2, pr):
    U = Solution.initialize(2*len(S1))
    u = np.array([crx_exponential(X1.x, X2.x, pr, crx_npoint) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i, Ui in enumerate(U[0::2]):
        Ui.setX(u[i,0])
        # Ui.copyStatusPSO(S1[i,0])
        Ui.setVelocity(S1[i,0].velocity)
        Ui.pbest = S1[i,0].pbest
        
    for i, Ui in enumerate(U[1::2]):
        Ui.setX(u[i,1])
        # Ui.copyStatusPSO(S2[i,0])
        Ui.setVelocity(S2[i,0].velocity)
        Ui.pbest = S2[i,0].pbest
    
    return U
    
# wrapper 1child
def w_crx_exp2(S1, S2, pr):
    U = Solution.initialize(len(S1))
    u = np.array([crx_exponential(X1.x, X2.x, pr, crx_npoint) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return U

# exchange points
def crx_npoint(x1, x2, points):
    u = np.array([_ for _ in x1])
    v = np.array([_ for _ in x2])
    
    u[points] = v[points]
    v[points] = u[points]
    
    return u, v

# choose points 
def crx_exponential(x1, x2, pr, func=crx_npoint):
    all_points = np.arange(x1.shape[0])
    i = np.random.choice(all_points)    #ensure at least one point
    crossover_points = [all_points[i]]

    while pr >= np.random.uniform(0, 1) and len(crossover_points) < len(all_points):
        i = (i+1) % len(all_points)
        crossover_points = crossover_points + [all_points[i]]
        
    
    u, v = func(x1, x2, crossover_points)
    return u, v

## MUTATION DE
# wrapper
def w_mut_de(S1, S2, S3, beta):
    U = Solution.initialize(len(S1))
    u = np.array([mut_de(X1.x, X2.x, X3.x, beta) for X1, X2, X3 in zip(S1[:,0], S2[:,0], S3[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i])
        # U[i].copyStatusPSO(S1[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return np.array(U)

# main
def mut_de(x1, x2, x3, beta):
    u = x1 + beta*(x2 - x3)
    return u

# MUTATION UNIFORM
#wrapper
def w_mut_uni(S, pr):
    U = Solution.initialize(len(S))
    u = np.array([mut_uniform(Xi.x, *Xi.bounds, pr) for Xi in S[:,0]])
    
    for i in range(len(U)): 
        U[i].setX(u[i])
        # U[i].copyStatusPSO(S[i,0])
        U[i].setVelocity(S[i,0].velocity)
        U[i].pbest = S[i,0].pbest
    
    return U

#base
def mut_uniform(x, lb, ub, pr):
    # u = [np.random.uniform(lb, ub) if np.random.random() < param['pr'] else x[i] for i in range(len(x))]
    
    u = [np.random.uniform(lb, ub) if np.random.random() < pr else x[i] for i in range(len(x))]
    
    return u

# PSO OPERATOR
#wrapper
def w_pso(S, w, c1, c2):
    U = Solution.initialize(len(S))
    v = np.array([pso_velocity(Xi.x, Xi.velocity, type(Xi).best.x, Xi.pbest['x'], w, c1, c2) for Xi in S[:,0]])
    u = np.array([pso_move(S[i,0].x, v[i]) for i in range(len(S))])
    
    for i in range(len(U)): 
        U[i].setX(u[i])
        U[i].setVelocity(v[i])
        U[i].pbest = S[i,0].pbest
    
    return U

#base
def pso_velocity(x, v, gbest, pbest, w, c1, c2):
    r1 = np.random.random(len(x))
    r2 = np.random.random(len(x))
    
    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    return v
    
def pso_move(x, v):
    u = x + v
    return u
    
## LEVY FLIGHT
# wrapper
def w_levy_flight(S):
    U = Solution.initialize(len(S))
    u = np.array([levy_flight(Xi.x, Xi.pbest['x']) for Xi in S[:,0]])
    # u = np.array([levy_flight(Xi.x, Xi.x) for Xi in S[:,0]])
    
    for i in range(len(U)): 
        U[i].setX(u[i])
        # U[i].copyStatusPSO(S[i,0])
        U[i].setVelocity(S[i,0].velocity)
        U[i].pbest = S[i,0].pbest
    
    return U

# base
beta = 3 / 2
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
def levy_flight(x, pbest):
    #beta and sigma are computed once the module is loaded
    
    w = np.array(np.random.standard_normal(x.shape)) * sigma
    v = np.array(np.random.standard_normal(x.shape))
    step = w / abs(v) ** (1 / beta)

    stepsize = 0.2 * step * (x - pbest)
    u = x + stepsize

    return u
    
### DROPOUT ###
## DROPOUT BY PROBABILITY
def drop_probability(X, pr):
    for i in range(len(X)):
        if np.random.random() < pr:
            X[i].setX(init_random(*type(X[i]).bounds, type(X[i]).dimension))
            
    return X

def drop_worst(X, pr, k):
    u = np.array([(X[i].getFitness(), i) for i in range(X.shape[0])])
    u = sorted(u, key=lambda x:x[0])
    for i in range(k):
        if np.random.random() < pr:
            ind = int(u[i][1])
            X[ind].setX(init_random(*type(X[ind]).bounds, type(X[ind]).dimension))
    return X        
           
### REPAIR OPERATOR ###
def repair_truncate(x, lb, ub):
    u = np.clip(x, lb, ub)
    return u
    
def repair_random(x, lb, ub):
    u = np.array([xi for xi in x])
    
    mask = (u<lb) + (u>ub)
    u[mask] = np.random.uniform(lb, ub, len(u[mask]))
    return u
    
def repair_reflect(x, lb, ub):
    u = np.array([xi for xi in x])
    
    mask_lb = u<lb
    mask_ub = u>ub
    
    u[mask_ub] = u[mask_ub]%ub
    u[mask_lb] = u[mask_lb]%lb

    return u

### KEEP-RULE / UPDATE-RULE ###
## REPLACE IF IMPROVED
def replace_if_best(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[i].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)


## REPLACE IF BETTER THAN A RANDOM - cuckoo-style update
def replace_if_random(X1, X2):
    U = [X2[i] if X2[i].getFitness() > X1[np.random.randint(0, X1.shape[0])].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)
    
    
### OTHER LEGACY FUNCTIONS ###
#auxilary function for op_de
def selection_for_op_de(X, sel):
    idx_tmp = np.arange(X.shape[0])
    idx = np.array([np.append([i], sel(X, np.delete(idx_tmp, i), 3, replace=False)) for i in range(X.shape[0])])

    return idx

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

#auxilary function to op_de
def apply_op_de(xi, xr1, xr2, xr3, mut, cross):
    u = mut(xr1, xr2, xr3)
    v, _ = cross(xi.x, u)
    return v