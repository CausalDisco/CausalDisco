import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC


def sort_regress(X, scores):
    """ 
    Regress each variable onto all predecessors in the ordering incurred by the criterion.
    Args:
        X: (n x d) matrix
        ordering: (n) vector 
    Returns:
        causal structure matrix with coefficients.
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
    Perform sort_regress using R^2 as ordering criterion. R^2 are computed using partial correlations obtained through matrix inversion.
    Args:
        X: n x d data,
    Returns:
        causal structure matrix with coefficients.
    """
    return sort_regress(X, 1 - np.diag(1/np.linalg.inv(np.corrcoef(X.T))))


if __name__ == "__main__":
    d = 10
    W = np.diag(np.ones(d-1), 1)
    print(f'True\n{W}')
    X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))
    X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    print('--- varSortnRegress ---')
    print(f'Recovered:\n{1.0*(var_sort_regress(X)!=0)}')
    print(f'Recovered standardized:\n{1.0*(var_sort_regress(X_std)!=0)}')

    print('--- r2SortnRegress ---')
    print(f'Recovered:\n{1.0*(r2_sort_regress(X)!=0)}')
    print(f'Recovered standardized:\n{1.0*(r2_sort_regress(X_std)!=0)}')