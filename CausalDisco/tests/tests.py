import pytest
import numpy as np


def test_order_alignment():
    from CausalDisco.analytics import order_alignment
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


def test_baselines():
    from scipy import linalg
    from CausalDisco.baselines import random_sort_regress, var_sort_regress, r2_sort_regress

    # generate data
    d = 10
    W = np.diag(np.ones(d-1), 1)
    X = np.random.randn(10000, d).dot(linalg.inv(np.eye(d) - W))
    X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # run baselines and print results
    print(
        f'True\n{W}\n'
        '--- randomRegress ---\n'
        f'Recovered:\n{1.0*(random_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(random_sort_regress(X_std)!=0)}\n'
        '--- varSortnRegress ---\n'
        f'Recovered:\n{1.0*(var_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(var_sort_regress(X_std)!=0)}\n'
        '--- r2SortnRegress ---\n'
        f'Recovered:\n{1.0*(r2_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(r2_sort_regress(X_std)!=0)}\n'
    )


def test_analytics():
    from scipy import linalg
    from CausalDisco.analytics import var_sortability, r2_sortability, snr_sortability

    # generate data
    d = 10
    W = np.diag(np.ones(d-1), 1)
    X = np.random.randn(10000, d).dot(linalg.inv(np.eye(d) - W))

    # run analytics and print results
    print(
        f'True\n{W}\n'
        f'var-sortability={var_sortability(X, W):.2f}\n'
        f'R^2-sortability={r2_sortability(X, W):.2f}\n'
        f'SNR-sortability={snr_sortability(X, W):.2f}'
    )
