import numpy as np
from sklearn.linear_model import LinearRegression


def order_alignment(W, scores, tol=0.):
    """
    Compute a measure for the agreement of an ordering incurred by the scores
    with a causal ordering incurred by the (weighted) adjacency matrix W.
    Args:
        X: (n x d) matrix
        W: (d x d) matrix
        scores: (d) vector
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

    # create d x d matrix of differences of scores such that
    # the entry in the i-th row and j-th column is
    #     * positive if score i < score j
    #     * zero if score i = score j
    #     * negative if score i > score j
    differences = scores - scores.T

    # measure ordering agreement
    # see 10.48550/arXiv.2102.13647, Section 3.1
    # and 10.48550/arXiv.2303.18211, Equation (3)
    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        # count 1/2 per correctly ordered pair and per unordered pair
        n_correctly_ordered_paths += (Ek * (differences >= 0 - tol)).sum() / 2
        # count another 1/2 per correctly ordered pair
        n_correctly_ordered_paths += (Ek * (differences > 0 + tol)).sum() / 2
        Ek = Ek.dot(E)
    return n_correctly_ordered_paths / n_paths


def var_sortability(X, W):
    return order_alignment(W, np.var(X, axis=0))


def r2_sortability(X, W):
    return order_alignment(W, np.diag(1 - 1/np.linalg.inv(np.corrcoef(X.T))))


def snr_sortability(X, W):
    d = X.shape[1]
    scores = np.zeros((1, d))
    LR = LinearRegression()
    for k in range(d):
        parents = W[:, k] != 0
        if np.sum(parents) > 0:
            LR.fit(X[:, parents], X[:, k])
            scores[0, k] = LR.score(X[:, parents], X[:, k])
    return order_alignment(W, scores)


def sanity_checks():
    d = 10
    W = np.diag(np.ones(d-1), 1)
    incr = np.arange(d, dtype='float')

    # correctly ordered
    assert order_alignment(W, incr) == 1., 'sanity check failed'
    # incorrectly ordered
    assert order_alignment(W, -incr) == 0., 'sanity check failed'
    # unordered
    assert order_alignment(W, np.zeros(d)) == .5, 'sanity check failed'
    assert order_alignment(W, np.ones(d)) == .5, 'sanity check failed'

    # ordered, yet very small scores
    # the difference between the smallest and largest score is (d-1)/1e5
    incr /= 1e5
    # with tol=0. no small increase is attributed to numerical imprecision
    assert order_alignment(W, incr, tol=0.) == 1., 'sanity check failed'
    # here all small increases are considered within numerical imprecision,
    # so all pairs are considered unordered
    assert order_alignment(W, incr, tol=(d-1)/1e5) == .5, 'sanity check failed'
    # if we slightly decrease the tolerance, some pairs are considered ordered
    # some unordered
    assert order_alignment(W, incr, tol=(d-2)/1e5) > .5, 'sanity check failed'
    assert order_alignment(W, incr, tol=(d-2)/1e5) < 1, 'sanity check failed'


if __name__ == "__main__":
    sanity_checks()

    d = 10
    W = np.diag(np.ones(d-1), 1)

    X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))
    print(f'True\n{W}')

    print(f'var-sortability={var_sortability(X, W):.2f}')
    print(f'R^2-sortability={r2_sortability(X, W):.2f}')
    print(f'SNR-sortability={snr_sortability(X, W):.2f}')
