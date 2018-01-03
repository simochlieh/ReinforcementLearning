import numpy as np
from tqdm import tqdm
import math

SQRT = np.vectorize(math.sqrt)


def run_lin_mab(model, alg_name, lambda_, alpha, epsilon, T, nb_simu):
    n_a = model.n_actions
    d = model.n_features
    regret = np.zeros((nb_simu, T))
    norm_dist = np.zeros((nb_simu, T))
    for k in tqdm(xrange(nb_simu), desc="Simulating {}".format(alg_name)):
        A = lambda_ * np.identity(d)
        b = np.zeros((d,))
        if epsilon == 0:
            beta = alpha * SQRT(np.diag(model.features.dot(1. / lambda_ * np.identity(d)).dot(model.features.T)))
        else:
            beta = 0.
        theta_hat = np.zeros((d,))
        for t in xrange(T):
            # algorithm that picks the action
            if np.random.rand(1) < epsilon:
                a_t = np.random.randint(0, n_a)
            else:
                a_t = np.argmax(model.features.dot(theta_hat) + beta)
            r_t = model.reward(a_t)  # get the reward
            # do something (update algorithm)
            # update theta_hat
            A = A + model.features[a_t, :].reshape(d, 1).dot(model.features[a_t, :].reshape(1, d))
            b = b + r_t * model.features[a_t, :]
            inv_A = np.linalg.inv(A)
            theta_hat = inv_A.dot(b)
            # update confidence bound
            if epsilon == 0:
                beta = alpha * SQRT(np.diag(model.features.dot(inv_A).dot(model.features.T)))
            # store regret
            regret[k, t] = model.best_arm_reward() - r_t
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    # # compute average (over sim) of the algorithm performance and plot it
    mean_norms = norm_dist.mean(axis=0)
    mean_regret = regret.mean(axis=0)

    return mean_norms, mean_regret
