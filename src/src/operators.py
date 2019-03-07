import numpy as np
from math import gamma, pi, sin
from .solution import *

### INITIALIZATION METHODS ###
def init_random(lb, ub, dimension):
    return np.random.uniform(lb, ub, dimension)
    
### SELECTION METHODS (INPUT) ###
def select_random(X, n=1):
    ''' Selects n exclusively and randomly candidate solutions from X for each Xi in X
    :param X: list of candidate solutions
    :param n: the number of individuals selected
    :returns: A list of references of the selected candidate solutions. '''
    S = [np.random.choice(X, n, replace=False) for _ in range(len(X))]
    return np.array(S)

def select_roulette(X, n=1):
    ''' Selects n exclusively candidate solutions from X through roulette for each Xi in X
    :param X: list of candidate solutions
    :param n: the number of individuals selected
    :returns: A list of references of the selected candidate solutions. '''
    #scale the fitness while keeping the original space between data
    f_scaled = scale([Xi.getFitness() for Xi in X], type(X[0]).maximize)
    #compute probability
    p = f_scaled/np.sum(f_scaled)
    #select
    S = [np.random.choice(X, size=n, replace=False, p=p) for i in range(len(X))]
    
    return np.array(S)
    
def select_tournament(X, n, k): 
    ''' Selects n exclusively candidate solutions from X for each Xi in X, through tournament of group size k
    :param X: list of candidate solutions
    :param n: the number of individuals selected
    :param k: group size
    :returns: A list of references of the selected candidate solutions. '''
    S = [np.sort(r)[::-1][:n] for r in select_random(X, k)]
    return np.array(S)

def select_current(X):
    ''' Selects all candidate solutions from X
    :param X: list of candidate solutions
    :returns: A list of references of the selected candidate solutions. '''
    S = [[Xi] for Xi in X]
    return np.array(S)

### OPERATORS ###  
## CROSSOVER BLEND
# wrapper_2children
def w_crx_blend(S1, S2, alpha):
    ''' 
    Blend Crossover Wrapper
        * status such as pso.velocity is copied from the solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param alpha: blend modifier
    :returns: list of blended candidate solutions
    '''
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
    ''' 
    Blend Crossover Wrapper
        * returns only one blended solution instead of two for each crossover
        * status such as pso.velocity is copied from the solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param alpha: blend modifier
    :returns: list of candidate solutions
    '''
    U = Solution.initialize(len(S1))
    u = np.array([crx_blend(X1.x, X2.x, alpha) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return U
    
# main
def crx_blend(x1, x2, alpha):
    ''' 
    Creates two new solutions by blending 'x1' and 'x2'
    based on deap's implementation
    (https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py)
    :param x1: np.array of real values
    :param x2: np.array of real values
    :param alpha: blend modifier
    :returns: (np.array, np.array)
    '''
    gamma = (1 + 2*alpha) * np.random.uniform(0, 1) - alpha
    u = (1 - gamma)*x1 + gamma*x2
    v = gamma*x1 + (1 - gamma)*x2
    
    return u, v
    
## CROSSOVER EXPONENTIAL
# wrapper 2children
def w_crx_exp(S1, S2, pr):
    ''' 
    Exponential Crossover Wrapper
        * status such as pso.velocity is copied from solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param alpha: blend modifier
    :returns: list of candidate solutions
    '''
    U = Solution.initialize(2*len(S1))
    u = np.array([crx_exponential(X1.x, X2.x, pr) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
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
    ''' 
    Exponential Crossover Wrapper
        * returns only the one resulted solution instead of two for each crossover
        * status such as pso.velocity is copied from solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param alpha: blend modifier
    :returns: list of candidate solutions
    '''
    U = Solution.initialize(len(S1))
    u = np.array([crx_exponential(X1.x, X2.x, pr) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return U
    
def crx_exponential(x1, x2, pr):
    ''' 
    Creates two new solutions by exchanging the points between 'x1' and 'x2', the points to be exchanged are chosen by the exponential method
    :param x1: np.array of real values
    :param x2: np.array of real values
    :returns: (np.array, np.array)
    '''
    size = len(x1)
    mask = [False]*size

    i = np.random.choice(size)
    mask[i] = True #ensures at least one point
    while pr >= np.random.uniform(0, 1) and np.sum(mask) < size:
        i = i + 1
        mask[i%size] = True

    u, v = crx_exchange_points(x1, x2, mask)

    return u, v

# exchange points
def crx_exchange_points(x1, x2, points):
    ''' 
    Creates two new solutions by exchanging the points between 'x1' and 'x2' based on the given 'points'
    :param x1: np.array of real values
    :param x2: np.array of real values
    :param points: list of indices
    :returns: (np.array, np.array)
    '''
    u = np.array([_ for _ in x1])
    v = np.array([_ for _ in x2])
    
    u[points] = x2[points]
    v[points] = x1[points]
    
    return u, v

# wrapper 2children
def w_crx_uni2(S1, S2, pr):    
    ## not implemented ##
    return
    
# wrapper 1child
def w_crx_uni2(S1, S2, pr):
    ''' 
    Exponential Crossover Wrapper
        * returns only the one resulted solution instead of two for each crossover
        * status such as pso.velocity is copied from solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param alpha: blend modifier
    :returns: list of candidate solutions
    '''
    U = Solution.initialize(len(S1))
    u = np.array([crx_exponential(X1.x, X2.x, pr) for X1, X2 in zip(S1[:,0], S2[:,0])])
    
    for i in range(len(U)):
        U[i].setX(u[i,0])
        U[i].setVelocity(S1[i,0].velocity)
        U[i].pbest = S1[i,0].pbest
    
    return U
    
# choose points 
def crx_uniform(x1, x2, pr): 
    ''' 
    Creates two new solutions by exchanging the points between 'x1' and 'x2', the points to be exchanged are chosen by the uniform method
    :param x1: np.array of real values
    :param x2: np.array of real values
    :returns: (np.array, np.array)
    '''
    size = len(x1)
    
    mask = [pr >= np.random.uniform(0, 1) for i in range(size)]
    
    #ensures one point
    if(np.sum(mask)==0):
        mask[np.random.choice(size)] = True

    u, v = crx_exchange_points(x1, x2, mask)
    return u, v

## MUTATION DE
# wrapper
def w_mut_de(S1, S2, S3, beta):
    ''' 
    DE Mutation Wrapper
        * status such as pso.velocity is copied from solutions in S1
    :param S1: list of selected candidate solutions
    :param S2: list of selected candidate solutions
    :param S3: list of selected candidate solutions
    :param beta: differential modifier
    :returns: list of candidate solutions
    '''
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
    ''' 
    Creates one new solutions by the differential mutation method
    :param x1: np.array of real values
    :param x2: np.array of real values
    :param x3: np.array of real values
    :returns: np.array of real values
    '''
    u = x1 + beta*(x2 - x3)
    return u

# MUTATION UNIFORM
#wrapper
def w_mut_uni(S, pr):
    ''' 
    Uniform Mutation Wrapper
    :param S: list of selected candidate solutions
    :param pr: probability
    :returns: list of candidate solutions
    '''
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
    ''' 
    Mutates the given candidate solution based on the probability 'pr' within the bounds 'lb' and 'ub'
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :param pr: probability
    :returns: np.array of real values
    '''
    u = [np.random.uniform(lb, ub) if np.random.random() <= pr else xi for xi in x]
    
    return np.array(u)

# PSO OPERATOR
#wrapper
def w_pso(S, w, c1, c2):
    ''' 
    PSO operator Wrapper
    :param S: list of selected candidate solutions
    :param w: (inertia) velocity modifier, real value
    :param c1: (cognitive) pbest modifier, real value
    :param c2: (social) gbest modifier, real value
    :returns: list of candidate solutions
    '''
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
    ''' 
    Computes the new velocity of 'x'
    :param x: np.array of real values
    :param v: np.array of real values
    :param gbest: np.array of real values
    :param pbest: np.array of real values
    :param w: (inertia) velocity modifier, real value
    :param c1: (cognitive) pbest modifier, real value
    :param c2: (social) gbest modifier, real value
    :returns: np.array of real values
    '''
    r1 = np.random.random(len(x))
    r2 = np.random.random(len(x))
    
    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    return v
    
def pso_move(x, v):
    ''' 
    Computes the new position of 'x'
    :param x: np.array of real values
    :param v: np.array of real values
    :returns: np.array of real values
    '''
    u = x + v
    return u
    
## LEVY FLIGHT
# wrapper
def w_levy_flight(S):
    ''' 
    Levy Flight (CS) Wrapper
    :param S: list of selected candidate solutions
    :returns: list of candidate solutions
    '''
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
    ''' 
    Computes the new position of 'x' through levy flight
    :param x: np.array of real values
    :param pbest: np.array of real values
    :returns: np.array of real values
    '''
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
    ''' 
    Uniformly drops candidate solutions in 'X', based on the probability 'pr', replacing their position by a uniform and random new position
    :param S: list of selected candidate solutions
    :param pr: probability of each solution being dropped
    :returns: list of candidate solutions
    '''
    for i in range(len(X)):
        if np.random.random() < pr:
            X[i].setX(init_random(*type(X[i]).bounds, type(X[i]).dimension))
            
    return X

def drop_worst(X, pr, k):
    ''' 
    Attempts to drop 'k' fitness-wise-worst solutions, based on the probability 'pr', replacing their position by a uniform and random new position
    :param S: list of selected candidate solutions
    :param pr: probability of each solution being dropped
    :param k: maximum number of candidate solutions to be dropped
    :returns: list of candidate solutions
    '''
    u = np.array([(X[i].getFitness(), i) for i in range(X.shape[0])])
    u = sorted(u, key=lambda x:x[0])
    for i in range(k):
        if np.random.random() < pr:
            ind = int(u[i][1])
            X[ind].setX(init_random(*type(X[ind]).bounds, type(X[ind]).dimension))
    return X        
           
### REPAIR OPERATOR ###
def repair_truncate(x, lb, ub):
    '''
    Replaces the values in 'x' higher than 'ub' and lower than 'lb' by 'ub' and 'lb', respectively
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :returns: np.array of real values
    '''
    u = np.clip(x, lb, ub)
    return u
    
def repair_random(x, lb, ub):
    '''
    Replaces the values in 'x' higher than 'ub' and lower than 'lb' by a random value between lb and ub
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :returns: np.array of real values
    '''
    u = np.array([xi for xi in x])
    
    mask = (u<lb) + (u>ub)
    u[mask] = np.random.uniform(lb, ub, len(u[mask]))
    return u
    
def repair_reflect(x, lb, ub):
    '''
    Replaces the values in 'x' higher than 'ub' and lower than 'lb' by value modulo 'ub' or 'lb', respectively 
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :returns: np.array of real values
    '''
    u = np.array([xi for xi in x])
    
    mask_lb = u<lb
    mask_ub = u>ub
    
    u[mask_ub] = u[mask_ub]%ub
    u[mask_lb] = u[mask_lb]%lb

    return u

### KEEP-RULE / UPDATE-RULE ###
## REPLACE IF IMPROVED
def replace_if_best(X1, X2):
    ''' 
    Compares the candidate solutions in X1 with the ones in X2 side by side - i.e., zip(X1, X2), returns the best ones based on their fitness
    :param X1: list of candidate solutions before a operator was applied
    :param X2: list of candidate solutions after a operator was applied
    :returns: list of candidate solutions
    '''
    U = [X2[i] if X2[i].getFitness() > X1[i].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)


## REPLACE IF BETTER THAN A RANDOM - cuckoo-style update
def replace_if_random(X1, X2):
    '''
    Compares each candidate solutions in X2 with a random one in X2, returns the ones in X2 if it is better than a random one or append the one in X1 with the same index if it is not.
    :param X1: list of candidate solutions before a operator was applied
    :param X2: list of candidate solutions after a operator was applied
    :returns: list of candidate solutions
    '''
    U = [X2[i] if X2[i].getFitness() > X2[np.random.randint(0, X2.shape[0])].getFitness() else X1[i] for i in range(X1.shape[0])]
    return np.array(U)
    
    
## AUXILIARY FUNCTIONS
def scale(x, maximize=False):
    '''
    Auxiliary function for roulette
    '''
    max_x = np.max(x)
    min_x = np.min(x)
    a = 1
    b = np.abs(min_x) + np.abs(max_x)
    
    if max_x != min_x: #convergence 
        result = np.array([a + (((xi - min_x)*(b-a))/(max_x-min_x)) for xi in x])
    else:
        result = np.array([1/len(x)]*len(x))
    
    if maximize:
        return result
    else:
        return (b+a) - result