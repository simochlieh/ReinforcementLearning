from gridworld import GridWorld1
import random
import gridrender as gui
import numpy as np
import matplotlib.pyplot as plt


NB_ACTIONS = 4


def matrix_to_array(q, env):
    """
    Remove zeros from numpy array and returns a list
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
    if random.randint(0, 100) < epsilon * 100:
        return np.random.choice(env.state_actions[state])
    else:
        return find_opt_action(env, state, q)


def find_opt_action(env, state, q):
    opt_action = env.state_actions[state][0]
    for a in xrange(q.shape[1]):
        if a in env.state_actions[state] and q[state, opt_action] < q[state, a]:
            opt_action = a

    return opt_action


def plot(x, y, title, xlabel, ylabel, num_fig):
    plt.figure(num_fig)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def get_q_and_j_estimates(env, p, nb_iter, t_max):
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
    state_action_counts = np.zeros((env.n_states, NB_ACTIONS + 1))

    # Initializing the matrix q
    q_estimate = np.zeros((env.n_states, NB_ACTIONS + 1))

    for i in xrange(nb_iter):
        if i % 1000:
            print "iteration %d" % i
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
    q = np.zeros((env.n_states, NB_ACTIONS))
    state_action_counts = np.zeros((env.n_states, NB_ACTIONS))
    infty_norm_errors = []
    cumulated_rewards =[]
    sum_rewards = 0
    p = None

    for n in xrange(nb_iter):
        if n % 1000 == 0:
            print "iteration %d" % n
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
        p = [find_opt_action(env, state, q) for state in xrange(env.n_states)]

        infty_norm_errors.append(max(np.array(v_opt) -
                                     np.array([q[state, p[state]] for state in xrange(env.n_states)])))
        cumulated_rewards.append(sum_rewards)

    return infty_norm_errors, cumulated_rewards, p


def main():

    # Initializing the environment model
    env = GridWorld1

    ################################################################################
    # Work to do: Q4
    ################################################################################
    # here the v-function and q-function to be used for question 4
    v_q4 = np.array([0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
                     -0.93358351, -0.99447514])
    q_q4 = [[0.87691855, 0.65706417],
            [0.92820033, 0.84364237],
            [0.98817903, -0.75639924, 0.89361129],
            [0.00000000],
            [-0.62503460, 0.67106071],
            [-0.99447514, -0.70433689, 0.75620264],
            [0.00000000],
            [-0.82847001, 0.49505225],
            [-0.87691855, -0.79703229],
            [-0.93358351, -0.84424050, -0.93896668],
            [-0.89268904, -0.99447514]
            ]
    # Optimal Policy: going right, and if not possible going up
    p_opt = [0 if 0 in state_action else 3 for state_action in env.state_actions]
    # gui.render_policy(env, p_opt)
    j_exact = v_q4.T.dot(np.ones(v_q4.shape[0]) * 1. / env.n_states)

    nb_iter = 10000
    t_max = 15
    q_estimate, j_estimate_values = get_q_and_j_estimates(env, p_opt, nb_iter, t_max)

    # q_list = matrix_to_array(q_estimate, env)
    # gui.render_q(env, q_list)
    # gui.render_q(env, q_q4)

    plot(range(1, nb_iter + 1), (np.array(j_estimate_values) - j_exact) / j_exact,
         title="Error between J estimated at iteration n and the exact J", xlabel="Number of iterations",
         ylabel="(J(n) - J) / J", num_fig=1)
    plt.close()

    ################################################################################
    # Work to do: Q5
    ################################################################################
    v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
             0.87691855, 0.82847001]

    # Let's first look at the effect of epsilon on the convergence
    nb_iter = 10000
    t_max = 10
    epsilon_values = np.arange(0., 1., 0.1)
    infinity_norm_errors_per_epsilon = []
    cumulated_rewards_per_epsilon = []

    for epsilon in epsilon_values:
        print "Epsilon value: %.2f" % epsilon
        infinity_norm_errors, cumulated_rewards, p = q_learning(env, nb_iter, t_max, epsilon, v_opt)
        infinity_norm_errors_per_epsilon.append(infinity_norm_errors[-1])
        cumulated_rewards_per_epsilon.append(cumulated_rewards[-1])

    plot(epsilon_values, infinity_norm_errors_per_epsilon, title="Infinity norm error w.r.t epsilon",
         xlabel="Epsilon", ylabel="Infinity norm error", num_fig=2)
    plot(epsilon_values, cumulated_rewards_per_epsilon, title="Cumulated reward w.r.t epsilon",
         xlabel="Epsilon", ylabel="Cumulated reward", num_fig=3)

    # Let's choose epsilon = 0.3 and see how the algorithm
    # converges w.r.t the number of iterations
    epsilon = 0.2
    print "Now Let's choose epsilon = %.2f" % epsilon
    infinity_norm_errors, cumulated_rewards, p = q_learning(env, nb_iter, t_max, epsilon, v_opt)
    plot(range(1, nb_iter + 1), infinity_norm_errors, title="Infinity norm error w.r.t the number of iterations",
         xlabel="Number of iterations", ylabel="Infinity norm error", num_fig=2)
    plot(epsilon_values, cumulated_rewards, title="Cumulated reward w.r.t the number of iterations",
         xlabel="Number of iterations", ylabel="Cumulated reward", num_fig=3)


if __name__ == '__main__':
    main()
