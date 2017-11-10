from gridworld import GridWorld1
import random
import numpy as np
import matplotlib.pyplot as plt

NB_POSSIBLE_ACTIONS = 4


def matrix_to_array(q, env):
    """
    Remove impossible (state, action) elements from numpy array q and returns a list q_list
    """
    result = []
    for i in xrange(q.shape[0]):
        result.append([])
        j = 0
        nb_actions = q.shape[1]
        while j < nb_actions:
            if j in env.state_actions[i]:
                result[i].append(q[i, j])
            j += 1

    return result


def take_greedy_action(env, epsilon, state, q):
    if random.randint(0, 99) < epsilon * 100:
        return np.random.choice(env.state_actions[state])
    else:
        return find_optimal_action(env, state, q)


def find_optimal_action(env, state, q):
    opt_action = env.state_actions[state][0]
    for a in xrange(q.shape[1]):
        if a in env.state_actions[state] and q[state, opt_action] < q[state, a]:
            opt_action = a

    return opt_action


def plot(x, y, title, xlabel, ylabel, num_fig, figsize=(15, 6)):
    plt.figure(num_fig, figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def get_monte_carlo_q_and_j_estimates(env, p, nb_iter, t_max):
    """
    Computing the Monte-Carlo estimate of q using the policy p
    and the value of J at each iteration n.

    :param env: GridWorld instance
    :param p: the policy we use at each step to compute q
    :param nb_iter: int
    :param t_max: int, maximum length of an episode
    """
    # To store the values of J at each iteration
    j_estimate_values = []

    # Initializing the matrix of counts for the pairs (state, action) when they occur
    state_action_counts = np.zeros((env.n_states, NB_POSSIBLE_ACTIONS + 1))

    # Initializing the matrix q
    q_estimate = np.zeros((env.n_states, NB_POSSIBLE_ACTIONS + 1))

    for i in xrange(nb_iter):
        if (i + 1) % (nb_iter / 10) == 0:
            print"iteration number: %d" % i
        t = 0

        # Choosing state0 and action0 randomly
        state0 = env.reset()
        action0 = np.random.choice(env.state_actions[state0])

        # Incrementing the count of (state0, action0)
        state_action_counts[state0, action0] += 1

        # Those variables store the state and action at time t
        state = state0
        action = action0
        term = False

        # Used to compute the incremental average
        return_trajectory = 0
        while t < t_max and not term:
            state, reward, term = env.step(state, action)
            action = p[state]
            return_trajectory += env.gamma ** t * reward
            t += 1
        q_estimate[state0, action0] = (1. - 1. / state_action_counts[state0, action0]) * q_estimate[state0, action0] + \
                                       1. / state_action_counts[state0, action0] * return_trajectory

        # We loop over states to compute j_estimate
        j_estimate = 0
        for state in xrange(env.n_states):
            j_estimate += 1. / env.n_states * q_estimate[state, p[state]]
        j_estimate_values.append(j_estimate)

    return q_estimate, j_estimate_values


def q_learning(env, nb_iter, t_max, epsilon, v_opt):
    """
    Implementing the q-learning algorithm
    :param env: GridWorld Instance
    :param nb_iter: int
    :param t_max: int
    :param epsilon: float between 0 and 1
    :param v_opt: the exact optimal value function
    :return: returns a list of infinity norm errors between v_opt and v_opt_estimated at every iteration,
             a list of cumlated reward at the end of every iteration,
             and the optimal policy found at the last iteration
    """
    # Initializing variables
    q = np.zeros((env.n_states, NB_POSSIBLE_ACTIONS))
    state_action_counts = np.zeros((env.n_states, NB_POSSIBLE_ACTIONS))
    infty_norm_errors = []
    cumulated_rewards =[]
    sum_rewards = 0
    p = None

    for n in xrange(nb_iter):
        if n % (nb_iter / 10) == 0:
            print "iteration number: %d" % n
        t = 0
        state0 = env.reset()
        term = False
        next_state = state0
        while not term and t < t_max:
            state = next_state

            # take greedy action with probability 1 - epsilon
            # and random action with probability epsilon
            action = take_greedy_action(env, epsilon, state, q)

            # incrementing count for (state, action)
            state_action_counts[state, action] += 1

            next_state, reward, term = env.step(state, action)
            sum_rewards += reward

            # Compute the temporal difference
            delta = reward + env.gamma * max(q[next_state, :]) - q[state, action]
            q[state, action] += 1. / state_action_counts[state, action] * delta

            t += 1

        # compute the greedy policy
        p = [find_optimal_action(env, state, q) for state in xrange(env.n_states)]

        infty_norm_errors.append(max(np.array(v_opt) -
                                     np.array([q[state, p[state]] for state in xrange(env.n_states)])))
        cumulated_rewards.append(sum_rewards)

    return infty_norm_errors, cumulated_rewards, p
