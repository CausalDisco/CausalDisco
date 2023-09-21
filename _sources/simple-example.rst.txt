Simple Example
----------------

Install package:

.. code-block:: bash

    $ pip install CausalDisco


Run analytics and baselines:

.. code-block:: python
    
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