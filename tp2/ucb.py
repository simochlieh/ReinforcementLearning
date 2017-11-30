import numpy as np
from tqdm import tqdm
import math

SQRT = np.vectorize(math.sqrt)


class UCB:

    def __init__(self, mab, t_max):
        self.mab = mab
        self.t_max = t_max
        self.k = len(self.mab)

        # matrix of numbers of draws of a at time t
        self.N = np.zeros((self.k, self.t_max))

        # matrix of sum of rewards gathered from pulling a up to time t
        self.S = np.zeros((self.k, self.t_max))
        self.rho = 0.2

    def UCB1(self):
        draws = []
        pulled_arm = None
        for t in tqdm(xrange(self.k), _desc='UCB1 initialization'):
            pulled_arm = t
            draws.append(pulled_arm)
            self.S[t, t] += pulled_arm.sample()
            self.N[t, t] += 1.

        empirical_mean = self.S[:, self.k] / self.N[:, self.k]
        for t in xrange(self.k, self.t_max):
            pulled_arm = np.argmax(empirical_mean + self.rho * math.sqrt(math.log(t) / 2.) *
                                   SQRT(1. / self.N[pulled_arm, t]))

            empirical_mean = self.S[:, t] / self.N[:, t]

