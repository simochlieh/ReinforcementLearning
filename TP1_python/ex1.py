import random
import numpy as np
from copy import deepcopy


class MDPModel:
    def __init__(self, p, r, gamma, set_actions, set_states, precision, max_iter=100):
        self.set_states = set_states
        self.n_states = len(set_states)
        self.set_actions = set_actions
        self.n_actions = len(set_actions)
        self.p = p
        assert [self.p.shape[i] for i in xrange(3)] == [self.n_states, self.n_states, self.n_actions], \
            "The shape of the probability matrix is (%s, %s, %s). This doesn't match the dimensions. " \
            "It should be (%s, %s, %s)" % (self.p.shape[0], self.p.shape[1], self.p.shape[2], self.n_states,
                                           self.n_states, self.n_actions)

        self.r = r
        assert [self.r.shape[i] for i in xrange(2)] == [self.n_states, self.n_actions], \
            "The shape of the reward matrix is (%s, %s). This doesn't match the dimensions. " \
            "It should be (%s, %s)" % (self.r.shape[0], self.r.shape[1], self.n_states, self.n_actions)

        self.gamma = gamma
        self.max_iter = max_iter
        self.precision = precision

        self.epsilon = precision * (1 - gamma) / (2 * gamma)
        self.policy_opt = np.zeros((self.n_states,))
        self.v_opt = None
        self.v_values = None
        self.last_iteration = None

    def step(self, state, action):
        """
        Computes the next state and the reward from the (state, action) pair
        :param state: int
        :param action: int
        :return: next_state, reward
        """
        random_nb = random.randint(0, 9)
        for next_state in xrange(self.n_states):
            if random_nb < self.p[next_state, state, action] * 10:
                return next_state, self.r[next_state, action]

    def apply_bellman(self):
        """
        Applies the bellman operator
        :param v: numpy array f shape(n_states,)
        """
        new_v = np.zeros((self.n_states,))
        for state in self.set_states:
            max_ = - float("inf")
            for action in self.set_actions:
                value = self.r[state, action] + self.gamma * self.p[:, state, action].dot(self.v_opt)
                if value > max_:
                    max_ = value
            new_v[state] = max_
        return new_v

    def compute_greedy_policy(self):
        """
        Returns the greedy policy at the given state with the given optimal value function
        :param state: int
        :param v_opt:
        """
        for state in self.set_states:
            max_ = - float("inf")
            argmax_action = None
            for action in self.set_actions:
                value = self.r[state, action] + self.gamma * self.p[:, state, action].dot(self.v_opt)
                if value > max_:
                    argmax_action = action
                    max_ = value
            self.policy_opt[state] = argmax_action

    def policy_evaluation(self, policy):
        proba_policy = np.diagonal(self.p[:, :, map(int, policy)].T).T
        return np.linalg.solve((np.identity(self.n_states) - self.gamma * proba_policy),
                               np.diagonal(self.r[:, map(int, policy)]))

    def run_value_iteration(self, v_0):
        self.v_values = []
        self.v_opt = deepcopy(v_0)
        for k in xrange(self.max_iter):
            if k % 10 == 0:
                print "Iteration number: %d" % k
            new_v = self.apply_bellman()
            error = max(new_v - self.v_opt)
            self.v_values.append(new_v)
            if error < self.epsilon:
                self.v_opt = new_v
                print "last iteration: %d" % k
                self.last_iteration = k
                break
            self.v_opt = new_v
        self.compute_greedy_policy()

    def run_policy_iteration(self, policy_0):
        self.policy_opt = deepcopy(policy_0)
        for k in xrange(self.max_iter):
            print "Iteration number: %d" % k
            new_v = self.policy_evaluation(self.policy_opt)
            if self.v_opt is not None and all(new_v == self.v_opt):
                break
            self.v_opt = new_v
            self.compute_greedy_policy()


