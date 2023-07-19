[![Latest version](https://badge.fury.io/py/CausalDisco.svg)](https://badge.fury.io/py/CausalDisco)
[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/CausalDisco?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/CausalDisco)


# CausalDisco 🪩

CausalDisco contains baseline algorithms and analytics tools for Causal Discovery. The [package](https://pypi.org/project/CausalDisco/) can be installed by running `pip install CausalDisco`. Additional information can be found in the [documentation](https://causaldisco.github.io/CausalDisco/).

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
- [Simple Sorting Criteria Help Find the Causal Order in Additive Noise Models](https://arxiv.org/abs/2303.18211).
```
@article{reisach2021beware,
  title={Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game},
  author={Reisach, Alexander G. and Seiler, Christof and Weichwald, Sebastian},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{reisach2023simple,
  title={Simple Sorting Criteria Help Find the Causal Order in Additive Noise Models},
  author={Reisach, Alexander G. and Tami, Myriam and Seiler, Christof and Chambaz, Antoine and Weichwald, Sebastian},
  journal={arXiv preprint arXiv:2303.18211},
  year={2023}
}
```

### A Simple Example
```python
## sample data from a linear SCM
import numpy as np
from CausalDisco.baselines import var_sort_regress, r2_sort_regress
from CausalDisco.analytics import var_sortability, r2_sortability

d = 10
W = np.diag(np.ones(d-1), 1)
X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# run analytics and print results
print(
    f'True\n{W}\n'
    f'var-sortability={var_sortability(X, W):.2f}\n'
    f'R^2-sortability={r2_sortability(X, W):.2f}\n'
    f'SNR-sortability={snr_sortability(X, W):.2f}'
)

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
```
