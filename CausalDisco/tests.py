import pytest
import numpy as np
from .analytics import order_alignment


def test_order_alignment():
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
