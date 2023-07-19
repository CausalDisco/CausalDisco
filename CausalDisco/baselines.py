import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression, LassoLarsIC
from CausalDisco.analytics import r2coeff


def sort_regress(X, scores):
    """
    Regress each variable onto all predecessors in
    the ordering implied by the scores.
    Args:
        X: (n x d) matrix
        scores: (d) vector
    Returns:
        Causal structure matrix with coefficients
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


def random_regress(X, seed=None):
    if seed is None:
        seed = np.random.randint(0, np.iinfo('int').max)
    rng = np.random.default_rng(seed)
    return sort_regress(X, rng.permutation(X.shape[1]))


def var_sort_regress(X):
    """
    Perform sort_regress using variances as ordering criterion.
    Args:
        X: n x d data,
    Returns:
        causal structure matrix with coefficients.
    """
    return sort_regress(X, np.var(X, axis=0))


def r2_sort_regress(X):
    """
    Perform sort_regress using R^2 as ordering criterion.
    Args:
        X: n x d data,
    Returns:
        causal structure matrix with coefficients.
    """
    return sort_regress(X, r2coeff(X.T))


if __name__ == "__main__":
    d = 10
    W = np.diag(np.ones(d-1), 1)
    X = np.random.randn(10000, d).dot(linalg.inv(np.eye(d) - W))
    X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    print(
        f'True\n{W}\n\n'
        '--- randomSortnRegress ---\n'
        f'Recovered:\n{1.0*(random_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(random_regress(X_std)!=0)}\n\n'
        '--- varSortnRegress ---\n'
        f'Recovered:\n{1.0*(var_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(var_sort_regress(X_std)!=0)}\n\n'
        '--- r2SortnRegress ---\n'
        f'Recovered:\n{1.0*(r2_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(r2_sort_regress(X_std)!=0)}'
    )
