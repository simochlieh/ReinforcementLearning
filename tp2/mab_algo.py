import numpy as np
from tqdm import tqdm


class MabAlgorithm(object):

    def __init__(self, mab, t_max, verbose):
        self.mab = mab
        self.t_max = t_max
        self.k = len(self.mab)

        # matrix of numbers of draws of a at time t
        self.N = np.zeros((self.k,))

        # matrix of sum of rewards gathered from pulling a up to time t
        self.S = np.zeros((self.k,))

        self.verbose = verbose

    def run(self):
        """
        Runs the algorithm for the input parameters

        :return:
            Sequence of rewards at each time t, Sequence of arms pulled at each time t
        """


class UCB(MabAlgorithm):

    def __init__(self, mab, t_max, rho, verbose=True):
        super(UCB, self).__init__(mab, t_max, verbose)
        self.rho = rho

    def run(self):
        # Sequence of pulled arms at time t
        draws = []

        # Sequence of rewards obtained at time t
        rew = []

        for t in tqdm(xrange(self.t_max), desc='UCB1 for t >= k', disable=self.verbose):
            if t < self.k:
                pulled_arm_idx = t
                draws.append(self.mab[pulled_arm_idx])
                reward = float(self.mab[pulled_arm_idx].sample())
                rew.append(reward)
                self.S[t] += reward
                self.N[t] += 1.
            else:
                pulled_arm_idx = np.argmax(self.S / self.N + self.rho * np.sqrt(np.log(t) / (2. * self.N)))
                draws.append(self.mab[pulled_arm_idx])
                reward = float(self.mab[pulled_arm_idx].sample())
                rew.append(reward)
                self.S[pulled_arm_idx] += reward
                self.N[pulled_arm_idx] += 1.

        return rew, draws


class ThompsonSampling(MabAlgorithm):

    def __init__(self, mab, t_max, verbose=True):
        super(ThompsonSampling, self).__init__(mab, t_max, verbose)

    def run(self):
        # Sequence of pulled arms at time t
        draws = []

        # Sequence of rewards obtained at time t
        rew = []

        theta = np.empty((self.k,))
        for _ in tqdm(xrange(self.t_max), desc='Thompson Sampling algorithm', disable=self.verbose):
            for k in xrange(self.k):
                theta[k] = self.mab[0].local_random.beta(self.S[k] + 1, self.N[k] - self.S[k] + 1)
            pulled_arm_idx = np.argmax(theta)
            draws.append(self.mab[pulled_arm_idx])
            reward = float(self.mab[pulled_arm_idx].sample())
            rew.append(reward)
            bernoulli_reward = int(self.bernoulli_trail(reward))

            self.S[pulled_arm_idx] += bernoulli_reward
            self.N[pulled_arm_idx] += 1.

        return rew, draws

    def bernoulli_trail(self, success_proba):
        return self.mab[0].local_random.rand(1) < success_proba


class NaiveStrategy(MabAlgorithm):

    def __init__(self, mab, t_max, verbose=True):
        super(NaiveStrategy, self).__init__(mab, t_max, verbose)

    def run(self):
        # Sequence of pulled arms at time t
        draws = []

        # Sequence of rewards obtained at time t
        rew = []

        pulled_arm_idx = np.random.randint(0, self.k)
        for _ in tqdm(xrange(self.t_max), desc='Naive Strategy', disable=self.verbose):
            draws.append(self.mab[pulled_arm_idx])
            reward = self.mab[pulled_arm_idx].sample()
            rew.append(reward)
            self.S[pulled_arm_idx] += reward
            self.N[pulled_arm_idx] += 1.
            empirical_mean_rewards = self.S / (self.N + 1e-5)
            pulled_arm_idx = np.argmax(empirical_mean_rewards)

        return rew, draws
