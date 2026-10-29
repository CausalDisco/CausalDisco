Simple Example
----------------

Install package:

.. code-block:: bash

    $ pip install CausalDisco


Run analytics and baselines:

.. code-block:: python
    
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

    # --- run baselines and print the structural Hamming distance (SHD),
    # i.e., the number of node pairs whose edge is missing, spurious, or reversed:

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
