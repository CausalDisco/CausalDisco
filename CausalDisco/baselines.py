import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC
from CausalDisco.analytics import r2coeff, rel_count_emp_hard


def sort_regress(X, scores, coef_tol=0.0):
    r"""
    Regress each variable onto all predecessors in
    the ordering implied by the scores.

    For each variable, the set of parents is selected by adaptive Lasso
    (``LassoLarsIC`` with BIC, weights from an initial OLS fit) on the
    standardized data, and the returned coefficients are OLS refits of the
    variable onto the selected parents on the centred, original-scale data.

    The selected support is invariant to rescaling the variables (exactly
    with sklearn's default noise-variance estimate in ``LassoLarsIC``, up to
    numerical precision). Standardization is kept for numerical conditioning,
    as sklearn no longer normalizes internally.

    Args:
        X: Data (:math:`n \times d` np.array).
        scores: Vector of scores (np.array with :math:`d` entries).
        coef_tol (optional): Adaptive Lasso coefficients (on the standardized scale) with absolute value at most coef_tol are treated as zero.

    Returns:
        Candidate causal structure matrix with coefficients
    """
    d = X.shape[1]
    W = np.zeros((d, d))
    ordering = np.argsort(scores)
    LR = LinearRegression(fit_intercept=False, copy_X=False)
    LL = LassoLarsIC(criterion="bic", fit_intercept=False, copy_X=False)
    # centering
    Xc = X - X.mean(axis=0, keepdims=True)
    # standardization for LARS (constant columns are left as zeros)
    sd = Xc.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.
    X_std = Xc / sd
    # backward regression
    for k in range(1, d):
        cov = ordering[:k]
        target = ordering[k]
        # initial OLS on standardized data
        LR.fit(X_std[:, cov], X_std[:, target])
        weight = np.abs(LR.coef_)
        # adaptive Lasso pruning on standardized data
        LL.fit(X_std[:, cov] * weight, X_std[:, target])
        pruned_idx = np.abs(LL.coef_ * weight) > coef_tol
        # refit OLS on centered original scale using selected predictors
        if pruned_idx.sum() > 0:
            selected_cov = cov[pruned_idx]
            LR.fit(Xc[:, selected_cov], Xc[:, target])
            W[selected_cov, target] = LR.coef_
    return W


def random_sort_regress(X, seed=None):
    r"""
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


def rel_sort_regress(X, alpha=0.05):
    r"""
    Perform sort_regress using the empirical number of relatives
    (see ``rel_count_emp_hard``) as ordering criterion.

    Args:
        X: Data (:math:`n \times d` np.array).
        alpha (optional): Significance level for the correlation test (default 0.05).

    Returns:
        Candidate causal structure matrix with coefficients.
    """
    return sort_regress(X, rel_count_emp_hard(X, alpha=alpha))
