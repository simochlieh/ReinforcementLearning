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


def get_mc_q_estimate(env, p, nb_iter, t_max):
    """
    Computing the Monte-Carlo estimate of q using the policy p

    :param env: GridWorld instance
    :param p: the policy we use at each step to compute q
    :param n_values: list of int, the values of n for which we compute J(n)
    :param t_max: int, maximum length of an episode
       """
    # Initializing the matrix of counts for the pairs (state, action) when they occur
    state_action_counts = np.zeros((env.n_states, NB_ACTIONS + 1))

    # Initializing the matrix q
    q_estimate = np.zeros((env.n_states, NB_ACTIONS + 1))

    for i in xrange(nb_iter):
        t = 0

        # Choosing state0 and action0 randomly
        state0 = env.reset()
        action0 = np.random.choice(env.state_actions[state0])

        term = False

        # Incrementing the count of (state0, action0)
        state_action_counts[state0, action0] += 1

        # Those variables store the state and action at time t
        state = state0
        action = action0

        # Used to compute the incremental average
        return_trajectory = 0
        while t < t_max and not term:
            state, reward, term = env.step(state, action)
            action = p[state]
            return_trajectory += env.gamma ** t * reward
            t += 1
        q_estimate[state0, action0] = (1. - 1. / state_action_counts[state0, action0]) * q_estimate[state0, action0] + \
                                       1. / state_action_counts[state0, action0] * return_trajectory
    # In case state_action_counts is 0 for some pairs,
    # q_estimate will also be equal to 0 (we never update q),
    # we add a very small term compared to state_action_counts which is >=1
    # in order to not divide by 0
    return q_estimate


def get_j_estimates(env, p, n_values, t_max):
    """
    Computing J for policy p for different values of n.
    This method also returns the last estimated value of q

    :param env: GridWorld instance
    :param p: the policy we use at each step to compute q
    :param n_values: list of int, the values of n for which we compute J(n)
    :param t_max: int, maximum length of an episode
    """
    j_estimate_values = []
    q_estimate = None

    for n in n_values:
        print "Computing J for n = %s" % n
        j_estimate = 0
        q_estimate = get_mc_q_estimate(env, p, n, t_max)

        # We loop over states to compute j_estimate
        for state in xrange(env.n_states):
            j_estimate += 1. / env.n_states * q_estimate[state, p[state]]
        j_estimate_values.append(j_estimate)

    return q_estimate, j_estimate_values


def main():

    # Initializing the environment model
    env = GridWorld1


    ################################################################################
    # investigate the structure of the environment
    # - env.n_states: the number of states
    # - env.state2coord: converts state number to coordinates (row, col)
    # - env.coord2state: converts coordinates (row, col) into state number
    # - env.action_names: converts action number [0,3] into a named action
    # - env.state_actions: for each state stores the action availables
    #   For example
    #       print(env.state_actions[4]) -> [1,3]
    #       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
    # - env.gamma: discount factor
    ################################################################################
    # print(env.state2coord)
    # print(env.coord2state)
    # print(env.state_actions)
    # print(env.action_names[env.state_actions[5]])

    ################################################################################
    # Policy definition
    # If you want to represent deterministic action you can just use the number of
    # the action. Recall that in the terminal states only action 0 (right) is
    # defined.
    # In this case, you can use gui.renderpol to visualize the policy
    ################################################################################
    # pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
    # gui.render_policy(env, pol)

    ################################################################################
    # Try to simulate a trajectory
    # you can use env.step(s,a, render=True) to visualize the transition
    ################################################################################

    # env.render = False
    # state = 0
    # fps = 1
    # for i in range(5):
    #     action = np.random.choice(env.state_actions[state])
    #     nexts, reward, term = env.step(state, action)
    #     state = nexts
    #     time.sleep(1. / fps)

    ################################################################################s
    # You can also visualize the q-function using render_q
    ################################################################################
    # first get the maximum number of actions available
    # max_act = max(map(len, env.state_actions))
    # q = np.random.rand(env.n_states, max_act)
    # gui.render_q(env, q)

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

    n_values = range(1, 1000, 10)
    t_max = 15
    q_estimate, j_estimate_values = get_j_estimates(env, p_opt, n_values, t_max)

    # q_list = matrix_to_array(q_estimate, env)
    # gui.render_q(env, q_list)
    # gui.render_q(env, q_q4)

    plot(n_values, np.array(j_estimate_values) - j_exact,
         title="Error between J estimated at iteration n and the exact J", xlabel="Number of iteration",
         ylabel="J(n) - J", num_fig=1)


    ################################################################################
    # Work to do: Q5
    ################################################################################
    # v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
    #          0.87691855, 0.82847001]
    #
    # nb_iter = 100000
    # t_max = 10
    # epsilon_values = [0.9]#np.arange(0.1, 1., 0.1)
    # q = np.zeros((env.n_states, NB_ACTIONS))
    # state_action_counts = np.zeros((env.n_states, NB_ACTIONS))
    # p_values = []
    # infty_norm_errors = []
    # infty_norm_errors_final = []
    # rewards_cumulated_values = []
    # rewards_cumulated_values_final = []
    # for epsilon in epsilon_values:
    #     for n in xrange(nb_iter):
    #         if n % 10000 == 0:
    #             print "iter n = %s" % n
    #         t = 0
    #         state0 = env.reset()
    #         term = False
    #         next_state = state0
    #         rewards_cumulated = 0
    #         while not term and t < t_max:
    #
    #             state = next_state
    #             action = take_greedy_action(env, epsilon, state, q)
    #
    #             # incrementing count for (state, action)
    #             state_action_counts[state, action] += 1
    #             next_state, reward, term = env.step(state, action)
    #             rewards_cumulated += reward
    #
    #             # Compute the temporal difference
    #             delta = reward + env.gamma * max(q[next_state, :]) - q[state, action]
    #             q[state, action] += 1. / state_action_counts[state, action] * delta
    #             t += 1
    #
    #         # compute the greedy policy
    #         p = [find_opt_action(env, state, q) for state in xrange(env.n_states)]
    #         p_values.append(p)
    #         infty_norm_errors.append(max(np.array(v_opt) -
    #                                      np.array([q[state, p[state]] for state in xrange(env.n_states)])))
    #
    #         rewards_cumulated_values.append(rewards_cumulated)
    #
    #     rewards_cumulated_values_final.append(sum(rewards_cumulated_values))
    #     infty_norm_errors_final.append(infty_norm_errors[-1])
    #     plt.figure(1)
    #     plt.title("Infinity norm error")
    #     plt.plot(range(1, nb_iter + 1), infty_norm_errors)
    #     plt.show()
    # print infty_norm_errors_final
    # plt.figure(2)
    # plt.title("Infinity norm error w.r.t epsilon")
    #
    # plt.plot(epsilon_values, infty_norm_errors_final)
    # plt.figure(3)
    # plt.title("Cumulated reward w.r.t epsilon")
    #
    # plt.plot(epsilon_values, rewards_cumulated_values_final)
    # plt.show()


if __name__ == '__main__':
     main()
