[![Latest version](https://badge.fury.io/py/CausalDisco.svg)](https://badge.fury.io/py/CausalDisco)
[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/CausalDisco?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/CausalDisco)


# CausalDisco 🪩

CausalDisco provides baseline algorithms and analytics tools for Causal Discovery in Python. The [package](https://pypi.org/project/CausalDisco/) can be installed by running `pip install CausalDisco`. Additional information can be found in the [documentation](https://causaldisco.github.io/CausalDisco/).

### Baseline Algorithms
Find the following baseline algorithms in __CausalDisco/baselines.py__
- R²-SortnRegress
- Var-SortnRegress
- Random-SortnRegress

### Analytics Tools
Find the following analytics tools in __CausalDisco/analytics.py__
- R²-sortability
- Var-sortability
- order_alignment

### Sources
If you find our algorithms useful please consider citing
- [Beware of the Simulated DAG!](https://proceedings.neurips.cc/paper_files/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Supplemental.pdf)
- [A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models](https://arxiv.org/abs/2303.18211).
```
@inproceedings{reisach2023scale,
    title = {{A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models}},
    author = {Alexander G. Reisach and Myriam Tami and Christof Seiler and Antoine Chambaz and Sebastian Weichwald},
    booktitle = {{Advances in Neural Information Processing Systems 36 (NeurIPS)}},
    year = {2023},
    doi = {10.48550/arXiv.2303.18211},
}

@inproceedings{reisach2021beware,
    title = {{Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game}},
    author = {Alexander G. Reisach and Christof Seiler and Sebastian Weichwald},
    booktitle = {{Advances in Neural Information Processing Systems 34 (NeurIPS)}},
    year = {2021},
    doi = {10.48550/arXiv.2102.13647},
}
```

### A Simple Example
```python
# --- sample data from a linear SCM:

import numpy as np
from scipy import linalg

d = 10
W = np.diag(np.ones(d-1), 1)
X = np.random.randn(10000, d).dot(linalg.inv(np.eye(d) - W))
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# --- run analytics and print results:

from CausalDisco.analytics import (
    var_sortability,
    r2_sortability,
    snr_sortability
)

print(
    f'True\n{W}\n'
    f'var-sortability={var_sortability(X, W):.2f}\n'
    f'R^2-sortability={r2_sortability(X, W):.2f}\n'
    f'SNR-sortability={snr_sortability(X, W):.2f}'
)

# --- run baselines and print results:

from CausalDisco.baselines import (
    random_sort_regress,
    var_sort_regress,
    r2_sort_regress
)

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
```
---
Shout out to [Anne Helby Petersen](https://github.com/annennenne) for coming up with the "causal disco" naming idea first (check out her [causalDisco](https://cran.r-project.org/web/packages/causalDisco/index.html) package providing tools for causal discovery on observational data in R).
