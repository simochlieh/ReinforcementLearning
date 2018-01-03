import numpy as np
import math
from tqdm import tqdm


def get_complexity(mab, mu_max):
    result = 0
    for arm in mab:
        if arm.mean < mu_max:
            result += (mu_max - arm.mean) / kl(arm.mean, mu_max)

    return result


def oracle(t, mab, mu_max):
    return get_complexity(mab, mu_max) * math.log(t)


def kl(x, y):
    return x * math.log(x / y) + (1. - x) * math.log((1. - x) / (1. - y))


def get_expected_regret(UCB1, TS, NS, T, nb_iter, mu_max):
    reg1 = np.zeros((nb_iter, T))
    reg2 = np.zeros((nb_iter, T))
    reg3 = np.zeros((nb_iter, T))
    for n in tqdm(xrange(nb_iter), desc='Running all 3 MAB algorithms for %d iterations' % nb_iter):
        rew1, draws1 = UCB1.run()
        reg1[n, :] = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
        rew2, draws2 = TS.run()
        reg2[n, :] = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)

        # reg3 = naive strategy

        rew3, draws3 = NS.run()
        reg3[n, :] = mu_max * np.arange(1, T + 1) - np.cumsum(rew3)
    mean_reg_UCB1 = reg1.mean(axis=0)
    mean_reg_TS = reg2.mean(axis=0)
    mean_reg_NS = reg3.mean(axis=0)

    return mean_reg_UCB1, mean_reg_TS, mean_reg_NS
