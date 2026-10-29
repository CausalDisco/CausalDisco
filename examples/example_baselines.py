"""Structure recovery by different SortnRegress variants baselines on raw and standardized data from a linear SCM."""
from CausalDisco.baselines import *
from CausalDisco.graphs import ER_DAG
from gadjid import shd
import numpy as np

# random DAG with 10 nodes and 2 expected edges per node,
# linear SCM with standard normal noise and edge weights in +-[0.5, 2]
rng = np.random.default_rng(0)
d = 10
B = ER_DAG(d, 2, shuffle=True, rng=rng)
W = B * rng.choice([-1, 1], size=(d, d)) * rng.uniform(.5, 2, size=(d, d))
print(f'True\n{B}')
X = rng.standard_normal((10000, d)).dot(np.linalg.inv(np.eye(d) - W))
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# structural Hamming distance: number of node pairs whose edge differs
# between the true and the estimated graph (missing, spurious, or reversed)
def SHD(W_hat):
    return shd(B.astype(np.int8), (W_hat != 0).astype(np.int8))[1]


for name, method in [
    ('randomSortnRegress', lambda X: random_sort_regress(X, seed=0)),
    ('varSortnRegress', var_sort_regress),
    ('r2SortnRegress', r2_sort_regress),
    ('relSortnRegress', rel_sort_regress),
]:
    print(f'--- {name} ---')
    print(f'SHD (raw data): {SHD(method(X))}')
    print(f'SHD (standardized data): {SHD(method(X_std))}')
