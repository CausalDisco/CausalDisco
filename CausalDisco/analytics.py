import numpy as np
from scipy import linalg
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from CausalDisco.graphs import reachability, count_relatives


### -------------------order alignment measures---------------------------------


def order_alignment_path_lengths(W, scores, tol=0.):
    r"""
    Fraction of ordered pairs :math:`(i, j)` whose scores increase in causal direction (:math:`s_i < s_j`, ties count :math:`1/2`), where each pair is counted once for every length :math:`k` such that a directed path of length :math:`k` from :math:`i` to :math:`j` exists. This is the definition of var- and :math:`R^2`-sortability in Reisach et al. 2021 (Section 3.1) and 2023 (Equation 3).

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


def order_alignment_edges(B, scores, tol=0.):
    r"""
    Fraction of edges :math:`i \to j` whose scores increase in causal direction (:math:`s_i < s_j`, ties count :math:`1/2`). Allows comparing across graph sizes and differentiates between alignment, randomness, and opposite order. With ties broken uniformly at random, :math:`(1 - v)|E|`, where :math:`v` is this measure and :math:`|E|` the number of edges, is the expected number of edges contradicting the order of the scores, and thus a lower bound on the expected Hamming distance between the graph and any graph compatible with that order (Reisach et al. 2026, Section 2.2).

    Args:
        B: Weighted/Binary DAG adjacency matrix (:math:`d \times d` np.array, ``B[i, j] != 0`` means :math:`i \to j`).
        scores: Vector of scores (np.array with :math:`d` entries).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).

    Returns:
        Scalar measure of agreement between the orderings (np.nan if B has no edges).
    """
    assert tol >= 0., 'tol must be non-negative'
    n_edges = np.sum(B != 0)
    if n_edges == 0:
        return np.nan
    # score_mat[i,j] = scores[j]-scores[i]: positive iff j scores higher than i
    score_mat = scores.reshape(1, -1) - scores.reshape(-1, 1)
    edge_scores = score_mat[B != 0]
    correct_score_pairs = np.sum(edge_scores > tol)
    equal_score_pairs = np.sum(np.abs(edge_scores) <= tol)
    return (correct_score_pairs + 0.5 * equal_score_pairs) / n_edges


def order_alignment_ancestors(B, scores, tol=0.):
    r"""
    Fraction of ancestor-descendant pairs :math:`(i, j)`, each counted once, whose scores increase in causal direction (:math:`s_i < s_j`, ties count :math:`1/2`). This is the path-existence sortability of Reisach et al. 2023 (Appendix D.1, Equation 5). It measures agreement with the partial order implied by the graph; if that order is total, it equals :math:`(1 + \tau_a)/2`, where :math:`\tau_a` is Kendall's tau-a between the scores and the causal order.

    Args:
        B: Weighted/Binary DAG adjacency matrix (:math:`d \times d` np.array, ``B[i, j] != 0`` means :math:`i \to j`).
        scores: Vector of scores (np.array with :math:`d` entries).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).

    Returns:
        Scalar measure of agreement between the orderings (np.nan if B has no edges).
    """
    R = reachability(B)
    np.fill_diagonal(R, False)
    return order_alignment_edges(R, scores, tol=tol)


### --------------helper functions for sortabilities----------------------------


def _corrcoef(X):
    """ np.corrcoef(X) (rows are variables), raising an error for constant variables, whose correlations are undefined. """
    constant = np.flatnonzero(np.ptp(X, axis=1) == 0)
    if len(constant) > 0:
        raise ValueError(f'constant variables {constant.tolist()} have undefined correlations, please remove them')
    return np.corrcoef(X)


def r2coeff(X):
    r"""
    Compute the :math:`R^2` of each variable using partial correlations obtained through matrix inversion.

    Args:
        X: Data (:math:`d \times n` np.array - note that the dimensions here are different from other methods, following np.corrcoef).

    Returns: 
        Array of :math:`R^2` values for all variables.
    """
    try:
        return 1 - 1/np.diag(linalg.inv(_corrcoef(X)))
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


def rel_count_emp_hard(X, alpha=0.05):
    r"""
    Estimate the number of relatives of each variable as the number of variables (including itself) whose correlation with it is significantly different from zero (two-sided t-test at level alpha).

    Args:
        X: Data (:math:`n \times d` np.array).
        alpha (optional): Significance level for the correlation test (default 0.05).

    Returns:
        Array of estimated relative counts for all variables.
    """
    corr = _corrcoef(X.T)
    n = len(X)
    df = n - 2
    tcrit = t.ppf(1 - alpha/2, df)
    cutoff = tcrit / np.sqrt(tcrit**2 + df)
    n_relatives = np.sum(np.abs(corr)>cutoff, axis=1)
    return n_relatives


### -------------------------sortability functions------------------------------


def _order_alignment(W, scores, tol, measure):
    """
    Agreement between the causal order implied by W and the order of the scores, i.e., the fraction of pairs (i, j) with s_i < s_j (ties count 1/2) among a set of causally ordered pairs determined by the measure:

    - "edges": every edge i -> j, counted once. Measures how well the scores orient the edges, and relates directly to the SHD of a sort-and-regress estimate (Reisach et al. 2026).
    - "ancestors": every ancestor-descendant pair, counted once. Measures how well the scores recover the causal (partial) order, irrespective of graph density.
    - "path_lengths": every ancestor-descendant pair, counted once for each distinct length of the directed paths connecting it. The definition of Reisach et al. 2021, 2023, kept to reproduce their results.
    """
    match measure:
        case "edges":
            return order_alignment_edges(W, scores, tol=tol)
        case "ancestors":
            return order_alignment_ancestors(W, scores, tol=tol)
        case "path_lengths":
            return order_alignment_path_lengths(W, scores, tol=tol)
        case _:
            raise ValueError(f'unknown measure {measure!r}, expected "edges", "ancestors", or "path_lengths"')


def var_sortability(X, W, tol=0., measure="edges"):
    r"""
    Sortability by variance.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Pairs over which the order alignment is computed. One of ``"edges"`` (default; see ``order_alignment_edges``), ``"ancestors"`` (see ``order_alignment_ancestors``), or ``"path_lengths"`` (the definition of Reisach et al. 2021, 2023; see ``order_alignment_path_lengths``).

    Returns:
        Var-sortability value (:math:`\in [0, 1]`) of the data
    """
    scores = np.var(X, axis=0)
    return _order_alignment(W, scores, tol, measure)


def r2_sortability(X, W, tol=0., measure="edges"):
    r"""
    Sortability by :math:`R^2`.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Pairs over which the order alignment is computed. One of ``"edges"`` (default; see ``order_alignment_edges``), ``"ancestors"`` (see ``order_alignment_ancestors``), or ``"path_lengths"`` (the definition of Reisach et al. 2021, 2023; see ``order_alignment_path_lengths``).

    Returns:
        :math:`R^2`-sortability value (:math:`\in [0, 1]`) of the data
    """
    scores = r2coeff(X.T)
    return _order_alignment(W, scores, tol, measure)


def snr_sortability(X, W, tol=0., measure="edges"):
    r"""
    Sortability by signal-to-noise (SnR) ratio (also referred to as cause-explained variance CEV).

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Pairs over which the order alignment is computed. One of ``"edges"`` (default; see ``order_alignment_edges``), ``"ancestors"`` (see ``order_alignment_ancestors``), or ``"path_lengths"`` (the definition of Reisach et al. 2021, 2023; see ``order_alignment_path_lengths``).

    Returns:
        SnR-sortability value (:math:`\in [0, 1]`) of the data
    """
    d = X.shape[1]
    scores = np.zeros((1, d))
    LR = LinearRegression()
    for k in range(d):
        parents = W[:, k] != 0
        if np.sum(parents) > 0:
            LR.fit(X[:, parents], X[:, k])
            scores[0, k] = LR.score(X[:, parents], X[:, k])
    return _order_alignment(W, scores, tol, measure)


def rel_sortability(X, W, tol=0., measure="edges", alpha=0.05):
    r"""
    Sortability by the empirical number of relatives (hard thresholding via correlation test).

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Pairs over which the order alignment is computed. One of ``"edges"`` (default; see ``order_alignment_edges``), ``"ancestors"`` (see ``order_alignment_ancestors``), or ``"path_lengths"`` (the definition of Reisach et al. 2021, 2023; see ``order_alignment_path_lengths``).
        alpha (optional): Significance level for the correlation test (default 0.05).

    Returns:
        rel-sortability value (:math:`\in [0, 1]`) of the data
    """
    scores = rel_count_emp_hard(X, alpha=alpha)
    return _order_alignment(W, scores, tol, measure)


### -------------------sortability by idealized criterion-----------------------


def rel_sortability_oracle(W, tol=0., measure="edges"):
    r"""
    Sortability by the true number of relatives as per W.

    Args:
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).
        tol (optional): Tolerance threshold for score comparisons (non-negative float).
        measure (optional): Pairs over which the order alignment is computed. One of ``"edges"`` (default; see ``order_alignment_edges``), ``"ancestors"`` (see ``order_alignment_ancestors``), or ``"path_lengths"`` (the definition of Reisach et al. 2021, 2023; see ``order_alignment_path_lengths``).

    Returns:
        Oracle rel-sortability value (:math:`\in [0, 1]`) of the graph
    """
    scores = count_relatives(W)
    return _order_alignment(W, scores, tol, measure)