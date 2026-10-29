"""Sortability of data from a linear SCM on a random DAG by variance, R^2, and number of relatives (all empirically estimated)."""
from CausalDisco.analytics import *
from CausalDisco.graphs import ER_DAG
import numpy as np

# random DAG with 10 nodes and 2 expected edges per node,
# linear SCM with standard normal noise and edge weights in +-[0.5, 2]
rng = np.random.default_rng(0)
d = 10
B = ER_DAG(d, 2, shuffle=True, rng=rng)
W = B * rng.choice([-1, 1], size=(d, d)) * rng.uniform(.5, 2, size=(d, d))
print(f'True\n{W.round(2)}')
X = rng.standard_normal((10000, d)).dot(np.linalg.inv(np.eye(d) - W))

print('Var-sortability=', var_sortability(X, W))
print('R^2-sortability=', r2_sortability(X, W))
print('rel-sortability=', rel_sortability(X, W))