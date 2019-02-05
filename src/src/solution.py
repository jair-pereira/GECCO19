import numpy as np

class solution(object):
    best = None
    history = []

    @staticmethod
    def updateBest(x):
        #special case:1st iteration (fix this later)
        if(solution.best == None or x.fitness > solution.best.fitness):
            solution.best = x
        return

    @staticmethod
    def updateHistory(X):
        solution.history = solution.history + [np.array([Xi.x for Xi in X])]
        return
        
    def __lt__(self, other):
        return self.getFitness() < other.getFitness()
        
    def __le__(self, other):
        return self.getFitness() <= other.getFitness()
        
    def __gt__(self, other):
        return self.getFitness() > other.getFitness()
        
    def __ge__(self, other):
        return self.getFitness() >= other.getFitness()

    def __init__(self, function, dimension, limits=(0,1)):
        self.x = np.zeros(dimension)

        self.function = function #static?
        
        self.fitness = None
        self.limits = limits #static?
        
        #pso attributes
        self.pbest_x = None
        self.pbest_f = None
        self.gbest   = None #static?
        self.velocity = np.zeros(dimension)
        
        self.age = 0
        self.rank = None
    
    def setX(self, x):
        self.x = np.clip(x, *self.limits)
        self.clearFitness()

    def getFitness(self):
        if self.fitness == None:
            self.fitness = self.evaluate()
            
            self.updatePBest()
        solution.updateBest(self)

        return self.fitness
    
    def clearFitness(self):
        self.fitness = None
    
    def evaluate(self):
        return -self.function(self.x)
        
    def initRandom(self):
        r = np.random.uniform(*self.limits, self.x.shape)
        self.setX(r)
        
    def updatePBest(self):
         #special case:1st iteration (fix this later)
        if(self.pbest_f == None or self.pbest_f > self.fitness):
            self.pbest_x = self.x
            self.pbest_f = self.fitness