[![Latest version](https://badge.fury.io/py/CausalDisco.svg)](https://badge.fury.io/py/CausalDisco)
[![Tests](https://github.com/CausalDisco/CausalDisco/actions/workflows/tests.yml/badge.svg)](https://github.com/CausalDisco/CausalDisco/actions/workflows/tests.yml)
[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/CausalDisco?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/CausalDisco)


# CausalDisco 🪩

CausalDisco provides baseline algorithms and analytics tools for Causal Discovery in Python. The [package](https://pypi.org/project/CausalDisco/) can be installed by running `pip install CausalDisco`. Additional information can be found in the [documentation](https://causaldisco.github.io/CausalDisco/).

### Functionality

**Baseline algorithms** can be found in [CausalDisco/baselines.py](https://github.com/CausalDisco/CausalDisco/blob/main/CausalDisco/baselines.py):
- Var-SortnRegress
- R²-SortnRegress
- rel-SortnRegress

**Analytics tools** can be found in [CausalDisco/analytics.py](https://github.com/CausalDisco/CausalDisco/blob/main/CausalDisco/analytics.py):
- Var-sortability
- R²-sortability
- rel-sortability (empirical and oracle number of relatives)

Each sortability can compute order alignment over edges, ancestor-descendant pairs, or path lengths.

**Random DAG models** can be found in [CausalDisco/graphs.py](https://github.com/CausalDisco/CausalDisco/blob/main/CausalDisco/graphs.py):
- Erdős–Rényi
- scale-free / Barabási–Albert
- time series: ER-based summary graphs, unrolled into DAGs

### Sources
If you find our algorithms useful please consider citing
1. [Beware of the Simulated DAG!](https://proceedings.neurips.cc/paper_files/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Supplemental.pdf)
2. [A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models](https://arxiv.org/abs/2303.18211)
3. [A Topological Sorting Criterion for Random Causal Directed Acyclic Graphs](https://arxiv.org/abs/2605.06288)
```
@inproceedings{reisach2026topological,
    title = {{A Topological Sorting Criterion for Random Causal Directed Acyclic Graphs}},
    author = {Alexander G. Reisach and Antoine Chambaz and Gilles Blanchard and Sebastian Weichwald},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2026},
    doi = {10.48550/arXiv.2605.06288},
}

@inproceedings{reisach2023scale,
    title = {A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models},
    author = {Alexander G. Reisach and Myriam Tami and Christof Seiler and Antoine Chambaz and Sebastian Weichwald},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2023},
    doi = {10.48550/arXiv.2303.18211},
}

@inproceedings{reisach2021beware,
    title = {Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game},
    author = {Alexander G. Reisach and Christof Seiler and Sebastian Weichwald},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2021},
    doi = {10.48550/arXiv.2102.13647},
}
```

### A Simple Example
```python
# --- sample data from a linear SCM on a random DAG:

import numpy as np
from CausalDisco.graphs import ER_DAG

rng = np.random.default_rng(0)
d = 10
B = ER_DAG(d, 2, shuffle=True, rng=rng)
W = B * rng.choice([-1, 1], size=(d, d)) * rng.uniform(.5, 2, size=(d, d))
X = rng.standard_normal((10000, d)).dot(np.linalg.inv(np.eye(d) - W))
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# --- run analytics and print results:

from CausalDisco.analytics import (
    var_sortability,
    r2_sortability,
    rel_sortability,
)

print(
    f'var-sortability={var_sortability(X, W):.2f}\n'
    f'R^2-sortability={r2_sortability(X, W):.2f}\n'
    f'rel-sortability={rel_sortability(X, W):.2f}'
)

# --- run baselines and print the structural Hamming distance (SHD)

from gadjid import shd

from CausalDisco.baselines import (
    random_sort_regress,
    var_sort_regress,
    r2_sort_regress,
    rel_sort_regress,
)

def SHD(W_hat):
    return shd(B.astype(np.int8), (W_hat != 0).astype(np.int8))[1]

for name, method in [
    ('randomSortnRegress', random_sort_regress),
    ('varSortnRegress', var_sort_regress),
    ('r2SortnRegress', r2_sort_regress),
    ('relSortnRegress', rel_sort_regress),
]:
    print(
        f'--- {name} ---\n'
        f'SHD: {SHD(method(X))}\n'
        f'SHD standardized: {SHD(method(X_std))}'
    )
```
---
Shout out to [Anne Helby Petersen](https://github.com/annennenne) for coming up with the "causal disco" naming idea (check out her [causalDisco](https://cran.r-project.org/web/packages/causalDisco/index.html) package providing tools for causal discovery on observational data in R).
