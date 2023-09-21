import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression


def order_alignment(W, scores, tol=0.):
    """
    Compute a measure for the agreement of an ordering incurred by the scores
    with a causal ordering incurred by the (weighted) adjacency matrix W.

    Args:
        W: :math:`(d x d)` matrix
        scores: (d) vector
        tol (optional): non-negative float

    Returns:
        Scalar measure of agreement between the orderings
    """
    assert tol >= 0., 'tol must be non-negative'
    E = W != 0
    Ek = E.copy()
    n_paths = 0
    n_correctly_ordered_paths = 0

    # arrange scores as row vector
    scores = scores.reshape(1, -1)

    # create d x d matrix of score differences of scores such that
    # the entry in the i-th row and j-th column is
    #     * positive if score i < score j
    #     * zero if score i = score j
    #     * negative if score i > score j
    differences = scores - scores.T

    # measure ordering agreement
    # see 10.48550/arXiv.2102.13647, Section 3.1
    # and 10.48550/arXiv.2303.18211, Equation (3)
    for _ in range(len(E) - 1):
        n_paths += Ek.sum()
        # count 1/2 per correctly ordered or unordered pair
        n_correctly_ordered_paths += (Ek * (differences >= 0 - tol)).sum() / 2
        # count another 1/2 per correctly ordered pair
        n_correctly_ordered_paths += (Ek * (differences > 0 + tol)).sum() / 2
        Ek = Ek.dot(E)
    return n_correctly_ordered_paths / n_paths


def r2coeff(X):
    """
    Compute the :math:`R^2` of each variable using partial correlations obtained through matrix inversion.

    Args:
        X: (d x n) array

    Returns: 
        Array of :math:`R^2` values for all variables
    """
    try:
        return 1 - np.diag(1/linalg.inv(np.corrcoef(X)))
    except linalg.LinAlgError:
        # fallback if correlation matrix is singular
        d = X.shape[0]
        r2s = np.zeros(d)
        LR = LinearRegression()
        X = X.T
        for k in range(d):
            parents = np.arange(d) != k
            LR.fit(X[:, parents], X[:, k])
            r2s[k] = LR.score(X[:, parents], X[:, k])
        return r2s


def var_sortability(X, W, tol=0.):
    r"""
    Sortability by variance. $x = 3$.
    
    Args:
        X: Data :math:`(n \times d)`
        W: Ground-truth :math:`(d \times d)` DAG adjacency matrix
    
    Returns:
        Var-sortability value (:math:`in [0, 1]`) of the data
    """
    return order_alignment(W, np.var(X, axis=0), tol=tol)


def r2_sortability(X, W, tol=0.):
    """
    Sortability by :math:`R^2`.
    
    Args:
        X: Data :math:`(n \times d)`
        W: Ground-truth :math:`(d \times d)` DAG adjacency matrix
    
    Returns:
        :math:`R^2`-sortability value (:math:`in [0, 1]`) of the data
    """
    return order_alignment(
        W,
        r2coeff(X.T),
        tol=tol)


def snr_sortability(X, W, tol=0.):
    """
    Sortability by signal-to-noise (SnR) ratio.

    Args:
        X: Data :math:`(n \times d)`
        W: Ground-truth :math:`(d \times d)` DAG adjacency matrix

    Returns:
        :math: SnR-sortability value (:math:`in [0, 1]`) of the data
    """
    d = X.shape[1]
    scores = np.zeros((1, d))
    LR = LinearRegression()
    for k in range(d):
        parents = W[:, k] != 0
        if np.sum(parents) > 0:
            LR.fit(X[:, parents], X[:, k])
            scores[0, k] = LR.score(X[:, parents], X[:, k])
    return order_alignment(W, scores, tol=tol)
