from scipy.spatial.distance import pdist, squareform
import cplex as cplex
import pandas as pd
import numpy as np
import itertools
import warnings


def get_preprocessing_space_and_distributions(samples_raw, samples_ctf):

    assert isinstance(samples_raw, pd.DataFrame)
    assert samples_raw.shape[0] > 0
    assert samples_raw.shape[1] > 0

    assert isinstance(samples_ctf, pd.DataFrame)
    assert samples_ctf.shape[0] > 0
    assert samples_ctf.shape[1] > 0

    assert np.all(samples_raw.columns == samples_ctf.columns)

    X_raw = np.array(samples_raw.as_matrix(), dtype = np.float_)
    X_ctf = np.array(samples_ctf.as_matrix(), dtype = np.float_)
    U = np.unique(np.vstack((X_raw, X_ctf)), axis = 0)

    N_raw = np.zeros(U.shape[0])
    N_ctf = np.zeros(U.shape[0])

    for i, u in enumerate(U):
        N_raw[i] = np.sum((X_raw == u).all(axis = 1))
        N_ctf[i] = np.sum((X_ctf == u).all(axis = 1))

    assert(np.sum(N_raw) == X_raw.shape[0])
    assert(np.sum(N_ctf) == X_ctf.shape[0])
    P = N_raw / np.sum(N_raw)
    Q = N_ctf / np.sum(N_ctf)

    assert np.isclose(1.0, np.sum(P))
    assert np.isclose(1.0, np.sum(Q))
    return U, P, Q


def build_transport_lp(distances, P, Q):
    n = distances.shape[0]
    assert np.all(distances >= 0.0)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diagonal(distances), np.zeros(n))

    # create cplex object
    cpx = cplex.Cplex()

    # add gamma variables to LP
    cpx.variables.add(names = ['gamma[%d][%d]' % (i, j) for i, j in itertools.product(list(range(n)), repeat = 2)],
                      obj = np.ravel(distances, order = 'C').tolist(),
                      lb = [0.0] * n ** 2,
                      ub = [1.0] * n ** 2)

    # add constraints of the form:
    # sum_j gamma[i][j] = p[i] for all i

    cvals = [1.0] * n
    cexpr = []
    for i in range(n):
        vnames = ['gamma[%d][%d]' % (i, j) for j in range(n)]
        cexpr += [cplex.SparsePair(ind = vnames, val = cvals)]

    cpx.linear_constraints.add(names = ['p[%d]' % i for i in range(n)],
                               lin_expr = cexpr,
                               senses = ['E'] * n,
                               rhs = P.tolist())

    # add constraints of the form:
    # sum_i gamma[i][j] = q[j] for all j

    cvals = [1.0] * n
    cexpr = []
    for j in range(n):
        vnames = ['gamma[%d][%d]' % (i, j) for i in range(n)]
        cexpr += [cplex.SparsePair(ind = vnames, val = cvals)]

    cpx.linear_constraints.add(names = ['q[%d]' % i for i in range(n)],
                               lin_expr = cexpr,
                               senses = ['E'] * n,
                               rhs = Q.tolist())

    # set parameters
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    cpx.set_problem_type(cpx.problem_type.LP)
    cpx.set_results_stream(None)
    cpx.parameters.randomseed.set(0)
    cpx.parameters.threads.set(0)
    cpx.parameters.output.clonelog.set(0)
    cpx.parameters.parallel.set(1)

    return cpx


def compute_preprocessing_distribution(U, P, Q):

    assert isinstance(U, np.ndarray)
    assert isinstance(P, np.ndarray)
    assert isinstance(Q, np.ndarray)
    assert U.ndim == 2

    P = P.flatten()
    Q = Q.flatten()

    n = U.shape[0]
    assert n > 1

    assert len(P) == n
    assert np.all((P > 0.0) & (P <= 1.0))
    assert np.isclose(np.sum(P), 1.0)

    assert len(Q) == n
    assert np.all((Q >= 0.0) & (Q <= 1.0))
    assert np.isclose(np.sum(Q), 1.0)

    distances = squareform(pdist(U, 'minkowski', p = 1.0))

    cpx = build_transport_lp(distances, P, Q)
    cpx.solve()
    gamma = cpx.solution.get_values()
    gamma = np.reshape(gamma, (n, n), 'C')
    T = gamma / P[:, None]

    # check that T is a distribution over the feature space
    assert T.shape[0] == n
    assert np.allclose(np.sum(T, axis = 1), 1.0)
    assert np.all(np.greater_equal(T, 0.0))
    return T


class RandomizedPreprocessor:

    # todo allow setting random state
    # todo vectorize lookup operations
    # todo vectorize sampling / uniform number generation
    # todo figure out what to do when x does not exist in U

    def __init__(self, U, processor_pdf, random_state = None):

        # check space
        assert isinstance(U, np.ndarray)
        assert U.ndim == 2
        n, d = U.shape
        assert n > 0
        assert d > 0
        assert n == np.unique(U, axis = 0).shape[0]

        # check pdf
        assert isinstance(processor_pdf, np.ndarray)
        assert processor_pdf.shape == (n, n)
        assert np.allclose(np.sum(processor_pdf, axis = 1), 1.0)
        assert np.all(np.greater_equal(processor_pdf, 0.0))
        assert np.all(np.isfinite(processor_pdf))

        # store processor cdf
        self.processor_pdf = processor_pdf
        self.processor_cdf = np.cumsum(processor_pdf, 1)
        self.U = np.array(U)
        self.n = n
        self.d = d

        # set random state
        self._random_state = None
        self.random_state = random_state


    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        if isinstance(seed, int):
            self._random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self._random_state = seed
        elif seed is None:
            self._random_state = np.random.RandomState()


    def get_index(self, x):
        """
        :param x: vector
        :return: index of point in U
        """
        pass


    def adjust_sample(self, x):
        u = np.random.random_sample()
        old_idx = np.flatnonzero((self.U == x).all(axis = 1))
        if len(old_idx) == 1:
            new_idx = np.argmin(self.processor_cdf[old_idx] < u)
            x_new = self.U[new_idx]
        else:
            # did not find x in U
            warnings.warn('did not find x in U')
            x_new = x

        return x_new


    def adjust(self, X):

        """
        :param X: matrix where each row is a sample
        :return: preprocessed matrix
        """
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[1] == self.d
        return np.vstack([self.adjust_sample(x) for x in X])

