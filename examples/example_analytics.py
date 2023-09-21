from CausalDisco.analytics import *
import numpy as np

d = 10
W = np.diag(np.ones(d-1), 1)
print(f'True\n{W}')
X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))

print('Var-sortability=', var_sortability(X, W))
print('R^2-sortability=', r2_sortability(X, W))
print('SNR-sortability=', snr_sortability(X, W))