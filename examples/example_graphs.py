"""Sampling of random ER, SF, and time-unrolled DAGs and the rel-sortability of linear SCM data on them."""
from CausalDisco.graphs import *
from CausalDisco.analytics import rel_sortability
import numpy as np

rng = np.random.default_rng(0)


def sample_linear_scm(B, n=10000):
    # linear SCM with standard normal noise and edge weights in +-[0.5, 1]
    d = len(B)
    W = B * rng.choice([-1, 1], size=(d, d)) * rng.uniform(.5, 1., size=(d, d))
    return rng.standard_normal((n, d)).dot(np.linalg.inv(np.eye(d) - W))


# random DAGs with 50 nodes and about 2 edges per node
for name, B in [
    ('ER', ER_DAG(50, 2, shuffle=True, rng=rng)),
    ('SF', SF_DAG(50, 2, shuffle=True, rng=rng)),
]:
    X = sample_linear_scm(B)
    print(f'{name}: {int(B.sum())} edges, '
          f'rel-sortability={rel_sortability(X, B):.2f}')

# time series: strongly connected summary graph over 10 variables with
# self-loops, unrolled over T time steps into a DAG with 10*T nodes
A_summary = sample_ER_summary_graph(10, edge_prob=.3, rng=rng)
for T in (5, 10, 20):
    B = time_unrolled_DAG(A_summary, T)
    X = sample_linear_scm(B)
    print(f'time series, T={T}: {int(B.sum())} edges, '
          f'rel-sortability={rel_sortability(X, B):.2f}')
