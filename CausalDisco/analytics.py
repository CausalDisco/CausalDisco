import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression


def order_alignment_paths(W, scores, tol=0.):
    r"""
    Computes a measure of the agreement between a causal ordering following the topology of the (weighted) adjacency matrix W and an ordering by the scores.

    Args:
        W: Weighted/Binary DAG adjacency matrix (:math:`d \times d` np.array).
        scores: Vector of scores (np.array with :math:`d` entries).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).

    Returns:
        Scalar measure of agreement between the orderings.
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


def order_alignment_adjacent_pairs(B, scores, tol=0.):
    """
    Sortability between adjacent pairs; allows comparing across graph sizes and differentiates between alignment, randomness, and opposite order. In the absence of draws, it can be used to derive a lower bound on the SHD (see Reisach et al. 2026, Section 2.2).

    Args:
        B (dxd np.array): binary adjacency matrix (B[i,j] != 0 means edge i -> j)
        scores (d np.array): real-valued evaluations of sortability criterion
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
    """
    assert tol >= 0., 'tol must be non-negative'
    n_edges = np.sum(B != 0)
    if n_edges == 0:
        return np.nan
    # score_mat[i,j] = scores[j]-scores[i]: positive iff j scores higher than i
    score_mat = scores.reshape(1, -1) - scores.reshape(-1, 1)
    edge_scores = (B * score_mat)[B != 0]
    correct_score_pairs = np.sum(edge_scores > tol)
    equal_score_pairs = np.sum(np.abs(edge_scores) <= tol)
    return (correct_score_pairs + 0.5 * equal_score_pairs) / n_edges



def r2coeff(X):
    r"""
    Compute the :math:`R^2` of each variable using partial correlations obtained through matrix inversion.

    Args:
        X: Data (:math:`d \times n` np.array - note that the dimensions here are different from other methods, following np.corrcoef).

    Returns: 
        Array of :math:`R^2` values for all variables.
    """
    try:
        return 1 - 1/np.diag(linalg.inv(np.corrcoef(X)))
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


def var_sortability(X, W, tol=0., measure="paths"):
    r"""
    Sortability by variance.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Order-alignment measure to use. One of ``"paths"`` (default) or ``"adjacent_pairs"``.

    Returns:
        Var-sortability value (:math:`\in [0, 1]`) of the data
    """
    scores = np.var(X, axis=0)
    match measure:
        case "paths":
            return order_alignment_paths(W, scores, tol=tol)
        case "adjacent_pairs":
            return order_alignment_adjacent_pairs(W, scores, tol=tol)


def r2_sortability(X, W, tol=0., measure="paths"):
    r"""
    Sortability by :math:`R^2`.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Order-alignment measure to use. One of ``"paths"`` (default) or ``"adjacent_pairs"``.

    Returns:
        :math:`R^2`-sortability value (:math:`\in [0, 1]`) of the data
    """
    scores = r2coeff(X.T)
    match measure:
        case "paths":
            return order_alignment_paths(W, scores, tol=tol)
        case "adjacent_pairs":
            return order_alignment_adjacent_pairs(W, scores, tol=tol)


def snr_sortability(X, W, tol=0., measure="paths"):
    r"""
    Sortability by signal-to-noise (SnR) ratio (also referred to as cause-explained variance CEV).

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Order-alignment measure to use. One of ``"paths"`` (default) or ``"adjacent_pairs"``.

    Returns:
        :math: SnR-sortability value (:math:`\in [0, 1]`) of the data
    """
    d = X.shape[1]
    scores = np.zeros((1, d))
    LR = LinearRegression()
    for k in range(d):
        parents = W[:, k] != 0
        if np.sum(parents) > 0:
            LR.fit(X[:, parents], X[:, k])
            scores[0, k] = LR.score(X[:, parents], X[:, k])
    match measure:
        case "paths":
            return order_alignment_paths(W, scores, tol=tol)
        case "adjacent_pairs":
            return order_alignment_adjacent_pairs(W, scores, tol=tol)
