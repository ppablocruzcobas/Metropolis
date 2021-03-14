# Metropolis-Hastings Random Walk Samplper algorithm

__author__ = 'Pedro Pablo'


import numpy as np


def density(x):
    return np.exp((-1 / 3) * (x[0]**2 - x[0] * x[1] + x[1]**2 + x[0] - 2 * x[1] + 1))


def mhrws(state, iters):
    states = [state]
    for i in range(iters):
        y = [states[i][0] + np.random.normal(0, 4),
             states[i][1] + np.random.normal(0, 4)]
        u = np.random.random()
        if u < min(1, density(y) / density(states[i])):
            states.append(y)
        else:
            states.append(states[i])
    return states


if __name__ == "__main__":
    ITERS = 5000
    states = mhrws([0, 0], ITERS)

    print("Estimated Value of E[X2] over %d iterations." % ITERS)
    print("E[X2] = %f" % (np.sum(states, axis=0) / ITERS)[1])
