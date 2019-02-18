import numpy as np
import pandas as pd
from scipy.special import expit as logit
from scipy.stats import binom

def to_array(v, n, dtype = float):
    """
    :param v: float, int, numpy.array / length with len 1 or d
    :param n: length of array
    :param dtype: target type
    :return:
    """
    assert v is not None
    n = int(n)

    if isinstance(v, list):
        v = np.array(v)
    elif isinstance(v, (int, float)):
        v = np.array([v])
    elif isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.flatten()

    if len(v) == 1:
        arr = np.repeat(v, n)
    else:
        arr = np.array(v)

    arr = arr.flatten().astype(dtype)
    assert arr.ndim == 1 and len(arr) == n
    return arr


def generate_toy_dataset(coefs, px, n_samples, group_label = 0):

    coefs = np.array(coefs, dtype = np.float).flatten()
    n_dim = len(coefs)
    assert n_dim >= 1

    if isinstance(px, float):
        px = np.array([px], dtype = np.float)
    elif isinstance(px, list):
        px = np.array(px, dtype = np.float)

    if len(px) == 1:
        px = np.repeat(px, n_dim)

    assert len(px) == n_dim
    qx = 1.0 - px

    # create function handles
    generate_x = lambda n: np.random.binomial(np.ones(n_dim, dtype = np.int), px, size = (n, n_dim)).astype(np.float)
    get_py = lambda x: logit(np.dot(x, coefs))
    simulate_uniform = lambda p: np.greater(p, np.random.uniform(0.0, 1.0, p.shape))
    generate_y = lambda x: 2.0 * simulate_uniform(get_py(x)) - 1.0
    get_y = lambda x: 2.0 * np.greater(get_py(x), 0.5) - 1.0

    def get_px(x):
        if x.ndim == 1:
            return np.prod(np.power(px, x) * np.power(qx, 1.0 - x))
        else:
            return np.prod(np.power(px, x) * np.power(qx, 1.0 - x), axis = 1)


    # build data frame
    x_names = ['x%d' % (j + 1) for j in range(n_dim)]
    x = generate_x(n_samples)
    y = generate_y(x)
    df = pd.DataFrame(x, columns = x_names)
    df.insert(0, 's', group_label)
    df.insert(0, 'y', y)

    handles = {'generate_x': generate_x,
               'generate_y': generate_y,
               'get_px': get_px,
               'get_py': get_py,
               'get_y': get_y}

    return df, handles


def generate_toy_dataset_binom(coefs, px, limits, n_samples, group_label = 0):

    coefs = np.array(coefs, dtype = np.float).flatten()
    n_dim = len(coefs)
    assert n_dim >= 1

    if isinstance(px, float):
        px = np.array([px], dtype = np.float)
    elif isinstance(px, list):
        px = np.array(px, dtype = np.float)

    if len(px) == 1:
        px = np.repeat(px, n_dim)

    assert len(px) == n_dim

    if isinstance(limits, (int, float)):
        limits = np.array([limits], dtype = np.int)
    elif isinstance(limits, list):
        limits = np.array(limits, dtype = np.int)

    if len(limits) == 1:
        limits = np.repeat(limits, n_dim)

    assert len(limits) == n_dim

    X = [binom(n = limits[i], p = px[i]) for i in range(n_dim)]
    generate_x = lambda n: np.vstack((x.rvs(n) for x in X)).transpose()

    def get_px(x):
        if x.ndim == 1:
            p = [p.pmf(x[i]) for i, p in enumerate(X)]
            return np.exp(np.sum(np.log(p)))
        else:
            p = [p.pmf(x[:, i]) for i, p in enumerate(X)]
            return np.exp(np.sum(np.log(p), axis = 0))


    get_py = lambda x: logit(np.dot(x, coefs))
    simulate_uniform = lambda p: np.greater(p, np.random.uniform(0.0, 1.0, p.shape))
    generate_y = lambda x: 2.0 * simulate_uniform(get_py(x)) - 1.0
    get_y = lambda x: 2.0 * np.greater(get_py(x), 0.5) - 1.0


    # build data frame
    x_names = ['x%d' % (j + 1) for j in range(n_dim)]
    x = generate_x(n_samples)
    y = generate_y(x)
    df = pd.DataFrame(x, columns = x_names)
    df.insert(0, 's', group_label)
    df.insert(0, 'y', y)

    handles = {'generate_x': generate_x,
               'generate_y': generate_y,
               'get_px': get_px,
               'get_py': get_py,
               'get_y': get_y}

    return df, handles
