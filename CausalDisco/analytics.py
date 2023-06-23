import numpy as np
from sklearn.linear_model import LinearRegression


def order_alignment(W, scores, tol=0.):
    """ 
    Compute a measure for the agreement of an ordering incurred by the scores with a causal ordering incurred by the (weighted) adjacency matrix W.
    Args:
        X: (n x d) matrix
        W: (d x d) matrix
        scores: (n) vector
    Returns:
        Scalar measure of agreement between the orderings
    """
    scores = scores.reshape(1, -1)
    E = W != 0
    Ek = E.copy()
    n_paths = 0
    n_correctly_ordered_paths = 0

    # translate to positive reals while keeping relative order intact
    if np.min(scores) <= tol:
        scores += tol - np.min(scores)

    # measure ordering agreement
    divide_err, invalid_err = np.geterr()['divide'], np.geterr()['invalid']
    np.seterr(divide='ignore', invalid='ignore')
    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * scores / scores.T > 1 + tol).sum()
        if tol > 0.:
            n_correctly_ordered_paths += 1/2*(
                (Ek * scores / scores.T <= 1 + tol) *
                (Ek * scores / scores.T >  1 - tol)).sum()
        Ek = Ek.dot(E)
    np.seterr(divide=divide_err, invalid=invalid_err)
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
            LR.fit(X[:, parents], X[:, k].ravel())
            scores[0, k] = LR.score(X[:, parents], X[:, k].ravel())
    return order_alignment(W, scores)



if __name__ == "__main__":
    d = 10
    W = np.diag(np.ones(d-1), 1)
    X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))
    print(f'True\n{W}')

    print('var-sortability=', var_sortability(X, W))
    print('R^2-sortability=', r2_sortability(X, W))
    print('SNR-sortability=', snr_sortability(X, W))