CausalDisco
===========

CausalDisco is distributed under the open source `3-clause BSD license
<https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE>`_.
If you publish work using CausalDisco, please consider citing the publications

- `Beware of the Simulated DAG! <https://proceedings.neurips.cc/paper_files/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Supplemental.pdf>`_ 
- `Simple Sorting Criteria Help Find the Causal Order in Additive Noise Models <https://arxiv.org/abs/2303.18211>`_.

.. code-block::

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


Installation
------------

.. code-block:: bash

    $ pip install CausalDisco


A Simple Example
----------------

.. code-block:: python
    
    ## sample data from a linear SCM
    import numpy as np
    d = 10
    W = np.diag(np.ones(d-1), 1)
    X = np.random.randn(10000, d).dot(np.linalg.inv(np.eye(d) - W))

    # use analytics tools
    from CausalDisco.analytics import var_sortability, r2_sortability
    print('var-sortability=', var_sortability(X, W))
    print('R^2-sortability=', r2_sortability(X, W))

    # run baselines
    from CausalDisco.baselines import var_sort_regress, r2_sort_regress
    X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    
    print('--- varSortnRegress ---')
    print(f'Recovered:\n{1.0*(var_sort_regress(X)!=0)}')
    print(f'Recovered standardized:\n{1.0*(var_sort_regress(X_std)!=0)}')
    
    print('--- r2SortnRegress ---')
    print(f'Recovered:\n{1.0*(r2_sort_regress(X)!=0)}')
    print(f'Recovered standardized:\n{1.0*(r2_sort_regress(X_std)!=0)}')
