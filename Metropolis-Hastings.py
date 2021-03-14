# Metropolis-Hastings algorithm

__author__ = 'Pedro Pablo'


import numpy as np


def is_distribution(p):
    return np.abs(np.sum(p) - 1.0) < 1e-8


def gen_disc_probability(vector):
    u = np.random.random()
    s = .0
    for i in range(len(vector) - 1):
        s = s + vector[i]
        if u < s:
            return i
    return len(vector) - 1


def prop_matrix(dim):
    """
    returns the proposition matrix Q of dimension `dim`.
    Q is an stochastic matrix (the sum over a row must be equal 1 always).
    The last part is achieved dividing each element in a row by the sum
    of that row.
    """
    # m = np.array([.5, .5, .5, .5])
    # return m.reshape((2, 2))
    m = np.random.random_sample((dim, dim))
    return m / m.sum(axis=1)[:, None]


def accept_matrix(Q, p, dim):
    """
    `p` is the final desired distribution.
    """
    a = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            a[j, i] = min((p[i] * Q[i, j]) / (p[j] * Q[j, i]), 1)
    return a


def m_hastings(state):
    temp_state = gen_disc_probability(Q[state, :])
    if temp_state == state:
        new_state = state
    else:
        u = np.random.random()
        if u < A.item((state, temp_state)):
            new_state = temp_state
        else:
            new_state = state
    return new_state


if __name__ == "__main__":
    ITERS = 5000  # number of iterations
    p = [.23, .77]  # distribution to simulate
    if is_distribution(p):
        Q = prop_matrix(len(p))
        A = accept_matrix(Q, p, len(p))
        # print(Q)
        # print(A)
        state = np.random.randint(len(p))  # random initial state
        states = [state]  # list of generated states
        for i in range(ITERS):
            state = m_hastings(state)
            if 5 * i > ITERS:  # only count last 1/5 of generated states
                states.append(state)
        print('%s Iterations. %s States.' % (ITERS, len(p)))
        print()
        # the next two lines have the intention to avoid numerical errors
        # in the simulated distribution
        v = [states.count(i) / len(states) for i in range(len(p))]
        eps = (1 - np.sum(v)) / len(p)
        for i in range(len(p)):
            v[i] += eps
            print('    P(X = %s) = %s' % (i, v[i]))
