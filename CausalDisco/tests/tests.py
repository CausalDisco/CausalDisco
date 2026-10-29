"""
Tests to be run from the root directory by `pytest CausalDisco/tests/tests.py`
"""
import numpy as np
import pytest


def test_order_alignment():
    from CausalDisco.analytics import order_alignment_path_lengths as order_alignment
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
    from CausalDisco.baselines import (
        random_sort_regress,
        var_sort_regress,
        r2_sort_regress,
        rel_sort_regress,
    )

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
        '--- relSortnRegress ---\n'
        f'Recovered:\n{1.0*(rel_sort_regress(X)!=0)}\n'
        f'Recovered standardized:\n{1.0*(rel_sort_regress(X_std)!=0)}\n'
    )

    # a constant column must not produce NaNs in sort_regress, but raises an
    # error for criteria based on correlations, which are undefined
    X_const = np.column_stack([X, np.ones(len(X))])
    assert np.all(np.isfinite(var_sort_regress(X_const))), 'NaN for constant column'
    for method in (r2_sort_regress, rel_sort_regress):
        with pytest.raises(ValueError, match='constant variables'):
            method(X_const)


def test_analytics():
    from scipy import linalg
    from CausalDisco.analytics import (
        var_sortability,
        r2_sortability,
        snr_sortability,
        rel_sortability_oracle,
        rel_sortability,
    )

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

    # v-structure 0->2, 1->2: count_relatives = [2, 2, 3]
    # both edges point toward the node with the highest count -> result = 1.0
    B = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=float)
    for measure in ('edges', 'ancestors', 'path_lengths'):
        assert rel_sortability_oracle(B, measure=measure) == 1.0, \
            f'v-structure should give 1.0 for measure={measure}'

    # sanity: both functions return values in [0, 1] on chain data
    for measure in ('edges', 'ancestors', 'path_lengths'):
        result = rel_sortability_oracle(W, measure=measure)
        assert 0. <= result <= 1., \
            f'rel_sortability_oracle out of range for measure={measure}'
        result = rel_sortability(X, W, measure=measure)
        assert 0. <= result <= 1., \
            f'rel_sortability out of range for measure={measure}'

    # edge weights (including negative ones) must not affect order alignment
    from CausalDisco.analytics import (
        order_alignment_edges,
        order_alignment_ancestors,
        order_alignment_path_lengths,
    )
    W_signed = W * np.where(np.arange(d) % 2, -2., .5)
    scores = np.random.randn(d)
    for fun in (order_alignment_edges, order_alignment_ancestors, order_alignment_path_lengths):
        assert fun(W_signed, scores) == fun(W, scores), \
            f'{fun.__name__} depends on edge weights'


def test_graphs():
    from CausalDisco.graphs import ER_DAG, SF_DAG, count_relatives

    d, c = 20, 2
    for sampler in (ER_DAG, SF_DAG):
        B = sampler(d, c, rng=0)
        assert np.array_equal(B, np.triu(B, k=1)), \
            f'{sampler.__name__} without shuffle should be upper triangular'
        B = sampler(d, c, shuffle=True, rng=0)
        assert np.all(np.linalg.matrix_power(B, d) == 0), \
            f'{sampler.__name__} with shuffle should be acyclic'
        assert np.array_equal(B, sampler(d, c, shuffle=True, rng=0)), \
            f'{sampler.__name__} should be reproducible given a seed'
    assert SF_DAG(d, c, rng=0).sum() == c * (d - c), 'SF_DAG should have c(d-c) edges'

    # chain 0 -> 1 -> 2 and isolated node 3
    B = np.zeros((4, 4))
    B[0, 1] = B[1, 2] = 1
    assert np.array_equal(count_relatives(B), [3, 3, 3, 1]), 'count_relatives failed'


def test_time_unrolling():
    from CausalDisco.graphs import (
        time_unrolled_DAG,
        sample_ER_summary_graph,
        is_strongly_connected,
        reachability,
    )

    rng = np.random.default_rng(0)
    p, T = 5, 4
    A = rng.binomial(1, .3, size=(p, p))
    np.fill_diagonal(A, 1)

    # reference: explicit loop over time steps
    B_loop = np.zeros((p * T, p * T))
    for t in range(T - 1):
        for i in range(p):
            for j in np.flatnonzero(A[i]):
                B_loop[t * p + i, (t + 1) * p + j] = 1
    B = time_unrolled_DAG(A, T)
    assert np.array_equal(B, B_loop), 'time_unrolled_DAG differs from loop'
    assert np.all(np.linalg.matrix_power(B, p * T) == 0), 'unrolled graph should be acyclic'

    # directed cycle is strongly connected, directed chain is not
    cycle = np.roll(np.eye(p), 1, axis=1)
    assert is_strongly_connected(cycle), 'cycle should be strongly connected'
    assert not is_strongly_connected(np.triu(cycle)), 'chain should not be strongly connected'

    # sampled summary graph is strongly connected with self-loops, so after
    # unrolling every variable at t is an ancestor of every variable at t + p - 1
    A_summary = sample_ER_summary_graph(p, edge_prob=.3, rng=0)
    assert np.all(np.diag(A_summary) == 1) and is_strongly_connected(A_summary), \
        'summary graph should be strongly connected with self-loops'
    B = time_unrolled_DAG(A_summary, T)
    assert reachability(B)[:p, (p - 1) * p:p * p].all(), 'ancestry claim failed'
