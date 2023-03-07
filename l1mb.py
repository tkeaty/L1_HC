import numpy as np
from sklearn.linear_model import lars_path
from scipy.stats import norm
import warnings


def compute_single_gaussian(x, mean):
    """
    This function assumes var=1
    :param x:
    :param mean:
    :return:
    """
    # Compute the constant
    const = 1/np.sqrt(2.0*np.pi)

    # Compute the exponential
    x_term = -0.5*((x - mean)**2)

    # Multiply the two terms together
    prob = const*np.exp(x_term)

    if prob == 0:
        print('x: %f, mean: %f' % (x, mean))

    return prob


def compute_log_gaussian_prob(mean, var, data):
    """
    Compute the log likelihood of the given univariate Gaussian data
    :param mean: sample mean for the feature
    :param var: sample variance for the feature
    :param data: Nx1 column vector of the observations for one feature
    :return: the log likelihood of the data
    """
    log_prob = 0
    warnings.filterwarnings('error')
    for d in data.values:
        try:
            log_prob += np.log(norm(mean, var).pdf(d))
        except Warning:
            print('x: %f, mean: %f' % (d, mean))
            log_prob += np.log(10 ** -30)

    return log_prob


def compute_log_gaussian_prob2(y_hat, y_true):
    """
    Compute the log likelihood of the observations given mean=y_hat, obs=y_true (used with product of linear regression)
    :param y_hat: Nx1 column vector corresponding to beta.T * X linear regression output
    :param y_true: Nx1 column vector corresponding to the observations of a given feature
    :return: the log likelihood of the data
    """
    log_prob = 0
    std = np.std(y_hat)

    warnings.filterwarnings('error')
    for h, t in zip(y_hat, y_true):
        try:
            log_prob += np.log(compute_single_gaussian(t, h))  # np.log(norm(h, std).pdf(t))
        except Warning:
            print('x: %f, mean: %f' % (t, h))
            log_prob += np.log(10 ** -30)

    return log_prob


def compute_max_log_linear_prob(X, y, W):
    """
    Compute the least-squares estimate of beta, for y=X*beta
    :param x_vec: indices of the covariate features for y
    :param y: row vector of observations
    :param data: pandas DataFrame
    :return: the log likelihood of the regression model P(y|X) = N(beta*x, var)
    """
    Y_hat = X @ W
    log_p = np.zeros(W.shape[1])

    for i, yh in enumerate(Y_hat.T):
        log_p[i] = compute_log_gaussian_prob2(yh, y)

    return np.argmax(log_p)


def get_path_discontinuities(coefs):
    inds = []
    last_disc = -1

    for i, alpha_coefs in enumerate(coefs.T):
        free_params = np.sum(np.where(np.abs(alpha_coefs) > 0, 1, 0))
        if free_params > last_disc:
            inds.append(i)
            last_disc = free_params

    return inds


def get_l1_markov_blanket(y, data):
    X = data.drop(y, axis=1)
    columns = X.columns
    X = X.values
    y_vec = data[y].values.flatten()

    _, _, coefs = lars_path(X, y_vec, method='lasso')
    path_discs = get_path_discontinuities(coefs)
    W = coefs[:, path_discs]

    best_alpha = compute_max_log_linear_prob(X, y_vec, W)
    col_set = np.where(np.abs(W[:, best_alpha].flatten()) > 0, True, False)

    return list(columns[col_set])


def initialize_pc_dict(mb_dict):
    pcs = {}

    for c, mb in mb_dict.items():
        for v in mb:
            if c not in pcs:
                pcs[c] = [v]
            elif v not in pcs[c]:
                pcs[c].append(v)

            if v not in pcs:
                pcs[v] = [c]
            elif c not in pcs[v]:
                pcs[v].append(c)

    return pcs


def get_l1mb_pcs(data):
    mb_dict = {}
    i = 0
    for c in data.columns:
        i += 1
        mb_dict[c] = get_l1_markov_blanket(c, data)

    return initialize_pc_dict(mb_dict)


