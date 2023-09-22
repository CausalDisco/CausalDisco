import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC
from CausalDisco.analytics import r2coeff


def sort_regress(X, scores):
    """
    Regress each variable onto all predecessors in
    the ordering implied by the scores.

    Args:
        X: Data (:math:`n \times d` np.array).
        scores: Vector of scores (np.array with :math:`d` entries).

    Returns:
        Candidate causal structure matrix with coefficients
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion='bic')
    d = X.shape[1]
    W = np.zeros((d, d))
    ordering = np.argsort(scores)

    # backward regression
    for k in range(1, d):
        cov = ordering[:k]
        target = ordering[k]
        LR.fit(X[:, cov], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, cov] * weight, X[:, target].ravel())
        W[cov, target] = LL.coef_ * weight
    return W


def random_sort_regress(X, seed=None):
    """
    Perform sort_regress using a random order.

    Args:
        X: Data (:math:`n \times d` np.array).
        seed (optional): random seed (integer)
    
    Returns:
        Candidate causal structure matrix with coefficients.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo('int').max)
    rng = np.random.default_rng(seed)
    return sort_regress(X, rng.permutation(X.shape[1]))


def var_sort_regress(X):
    r"""
    Perform sort_regress using variances as ordering criterion.

    Args:
        X: Data (:math:`n \times d` np.array).
    
    Returns:
        Candidate causal structure matrix with coefficients.
    """
    return sort_regress(X, np.var(X, axis=0))


def r2_sort_regress(X):
    r"""
    Perform sort_regress using :math:`R^2` as ordering criterion.

    Args:
        X: Data (:math:`n \times d` np.array).
    
    Returns:
        Candidate causal structure matrix with coefficients.
    """
    return sort_regress(X, r2coeff(X.T))
