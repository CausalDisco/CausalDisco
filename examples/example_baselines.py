from CausalDisco.baselines import *
import numpy as np

d = 10
W = np.diag(np.ones(d-1), 1)
print(f'True\n{W}')
X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

print('--- varSortnRegress ---')
print(f'Recovered:\n{1.0*(var_sort_regress(X)!=0)}')
print(f'Recovered standardized:\n{1.0*(var_sort_regress(X_std)!=0)}')

print('--- r2SortnRegress ---')
print(f'Recovered:\n{1.0*(r2_sort_regress(X)!=0)}')
print(f'Recovered standardized:\n{1.0*(r2_sort_regress(X_std)!=0)}')