import numpy as np
import copy
    
class Solution(object):
    #function related
    maximize  = None
    function  = None
    bounds    = ()
    dimension = None
    nfe       = 0
    #common attributes
    best      = None
    worst     = None
    gbest     = best
    history   = []
    
    @staticmethod
    def setProblem(function, bounds, dimension, maximize):
        Solution.maximize  = maximize
        Solution.function  = function
        Solution.bounds    = bounds
        Solution.dimension = dimension
        Solution.nfe       = 0
        
        Solution.sign = 1 if Solution.maximize else -1
    
    @staticmethod
    def initialize(n):
        return np.array([Solution() for i in range(n)])

    def __init__(self):
        #default attributes
        self.x = np.zeros(Solution.dimension)
        self.fitness  = None
        #pso attributes
        self.pbest    = {}#{'x': None, 'fitness': None}
        self.velocity = np.zeros(Solution.dimension)
        #bee attributes
        self.age  = 0
        self.rank = None
        
    def setX(self, x):
        # self.x = x
        self.x = np.clip(x, *self.bounds)
        self.clearFitness()

    def getFitness(self):
        if self.fitness == None:
            self.fitness = self.evaluate()
            Solution.updateBest(self)
            Solution.updateWorst(self)
            self.updatePBest()
        return self.fitness
    
    def clearFitness(self):
        self.fitness = None
    
    def evaluate(self):
        Solution.nfe += 1
        return (Solution.sign) * Solution.function(self.x)
        
    # PSO
    def updatePBest(self):
        if(self.pbest == {} or self.fitness >= self.pbest['fitness']):
            self.pbest['x']       = self.x
            # self.pbest['fitness'] = self.getFitness()
            self.pbest['fitness'] = self.fitness
    
    # Bee        
    def increaseAge(self):
        self.age+=1
    
    @staticmethod
    def updateBest(x):
        if(Solution.best == None or x >= Solution.best):
            Solution.best  = copy.deepcopy(x)
            Solution.gbest = Solution.best
        return
        
    @staticmethod
    def updateWorst(x):
        if(Solution.worst == None or x <= Solution.worst):
            Solution.worst  = copy.deepcopy(x)
        return
  
    @staticmethod
    def updateHistory(X):
        Solution.history = Solution.history + [np.array([Xi.x for Xi in X])]
        return
        
    @staticmethod
    def print(sep="\n"):
        output = "--------------------------------------------"+sep \
        +"FUNCTION:   "+str(Solution.function)         +sep \
        +"MAXIMIZING: "+str(Solution.maximize)         +sep \
        +"BOUNDS:     "+str(Solution.bounds)           +sep \
        +"DIMENSION:  "+str(Solution.dimension)        +sep \
        +"NFE:        "+str(Solution.nfe)              +sep \
        +"BEST FITNESS:  "+str(Solution.best.fitness)  +sep \
        +"WORST FITNESS: "+str(Solution.worst.fitness) +sep \
        +"--------------------------------------------"  
        print(output)
    
    #compare by fitness
    def __lt__(self, other):
        return (Solution.sign) * self.getFitness() < (Solution.sign) * other.getFitness()
        
    def __le__(self, other):
        return (Solution.sign) * self.getFitness() <= (Solution.sign) * other.getFitness()
    
    def __gt__(self, other):
        return (Solution.sign) * self.getFitness() > (Solution.sign) * other.getFitness()
     
    def __ge__(self, other):
        return (Solution.sign) * self.getFitness() >= (Solution.sign) * other.getFitness()