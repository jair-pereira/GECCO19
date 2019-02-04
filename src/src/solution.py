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

    def __init__(self, function, dimension, limits=(0,1)):
        self.x = np.zeros(dimension)

        self.function = function #static?
        
        self.fitness = None
        self.limits = limits #static?
        
        self.pbest = None
        self.gbest = None #static?
        self.velocity = np.zeros(dimension)
        self.age = 0
        self.rank = None
    
    def setX(self, x):
        self.x = x
        self.clearFitness()

    def getFitness(self):
        if self.fitness == None:
            self.fitness = self.evaluate()
            solution.updateBest(self)
        if(self.pbest == None or self.fitness > evaluate(self.pbest):
            self.pbest = self.x
        self.gbest = solution.best.x
        return self.fitness
    
    def clearFitness(self):
        self.fitness = None
    
    def evaluate(self):
        return -self.function(self.x)
        
    def initRandom(self):
        r = np.random.uniform(*self.limits, self.x.shape)
        self.setX(r)
        self.getFitness()
