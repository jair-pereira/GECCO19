from math import gamma, pi, sin
import numpy as np
from random import normalvariate, randint, random

class sw(object):

    def __init__(self):

        self.__Positions = []
        self.__Gbest = []
        self.__Nests = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)

    def _newbee(self, l, c, lb, ub):

        bee = []
        for i in l:
            new = [self._neighbor(i, lb, ub) for k in range(c)]
            bee += new
        bee += l

        return bee

    def _neighbor(self, who, lb, ub):

        neighbor = np.array(who) + np.random.uniform(-1, 1) * (
            np.array(who) - np.array(
                self._agents[np.random.randint(0, len(self._agents) - 1)]))
        neighbor = np.clip(neighbor, lb, ub)

        return list(neighbor)

    def _velocity(self, Pbest, Gbest, n, dimension, velocity):
        r1 = np.random.random((n, dimension))
        r2 = np.random.random((n, dimension))
        velocity = 0.5 * velocity + 1 * r1 * (Pbest - self._agents) + 1 * r2 * (Gbest - self._agents)
        self._agents += velocity

    def _Levyfly(self, Pbest, n, dimension):

        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (
            gamma((1 + beta) / 2) * beta *
            2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.array([normalvariate(0, 1) for k in range(dimension)]) * sigma
        v = np.array([normalvariate(0, 1) for k in range(dimension)])
        step = u / abs(v) ** (1 / beta)

        for i in range(n):
            stepsize = 0.2 * step * (self._agents[i] - Pbest)
            self._agents[i] += stepsize * np.array([normalvariate(0, 1)
                                                    for k in range(dimension)])
    
    def _drop_worst_chance(self, nest, lb, ub, dimension, function):
        fnests = [(function(self._nests[i]), i) for i in range(nest)]
        fnests.sort()
        nworst = nest // 2
        worst_nests = [fnests[-i - 1][1] for i in range(nworst)]

        for i in worst_nests:
            if random() < 0.25:
                self._nests[i] = np.random.uniform(lb, ub, (1, dimension))

    def _ordered_swap(self, n, nest, function):
        fnests = [(function(self._nests[i]), i) for i in range(nest)]
        fnests.sort()
        fcuckoos = [(function(self._agents[i]), i) for i in range(n)]
        fcuckoos.sort(reverse=True)
        if nest > n:
            mworst = n
        else:
            mworst = nest

        for i in range(mworst):
            if fnests[i][0] < fcuckoos[i][0]:
                self._agents[fcuckoos[i][1]] = self._nests[fnests[i][1]]