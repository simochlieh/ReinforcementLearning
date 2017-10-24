from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time
import math


def main():
    print "start"
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
    print(env.state2coord)
    print(env.coord2state)
    print(env.state_actions)
    print(env.action_names[env.state_actions[5]])

    ################################################################################
    # Policy definition
    # If you want to represent deterministic action you can just use the number of
    # the action. Recall that in the terminal states only action 0 (right) is
    # defined.
    # In this case, you can use gui.renderpol to visualize the policy
    ################################################################################
    pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
    gui.render_policy(env, pol)

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
    max_act = max(map(len, env.state_actions))
    # q = np.random.rand(env.n_states, max_act)
    # gui.render_q(env, q)

    ################################################################################
    # Work to do: Q4
    ################################################################################
    # here the v-function and q-function to be used for question 4
    v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
            -0.93358351, -0.99447514]
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

    # Loop over th number of outer iterations n that we choose:
    n = 50
    # Policy: going right, and if not possible going up
    p = [0 if 0 in state_action else 3 for state_action in env.state_actions]
    gui.render_policy(env, p)

    Tmax = 20
    # Matrix of number of times the pair (state, action) occured
    N = np.zeros((env.n_states, max_act + 1))
    # initializing q
    q = np.zeros((env.n_states, max_act + 1))
    for i in xrange(n):
        print "step %s" % i
        t = 0
        state0 = env.reset()
        action0 = p[state0]
        term = False
        N[state0, action0] += 1
        next_step = state0
        next_action = action0
        while t < Tmax or not term:
            next_step, reward, term = env.step(next_step, next_action)
            next_action = p[next_step]
            q[state0, action0] += env.gamma ** t * reward
            t += 1

    q = q / (N + 1e-2)
    gui.render_q(env, q)

    ################################################################################
    # Work to do: Q5
    ################################################################################
    v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
             0.87691855, 0.82847001]

if __name__ == '__main__':
    main()
