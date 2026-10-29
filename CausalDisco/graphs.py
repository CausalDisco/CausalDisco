import numpy as np


### -------------------------random DAG models----------------------------------


def _permute_nodes(B, rng):
    perm = rng.permutation(len(B))
    return B[perm][:, perm]


def ER_DAG(d, c, shuffle=False, rng=None):
    r"""
    Erdős–Rényi DAG with :math:`cd` expected edges: each :math:`i \to j`, :math:`i < j`, independently with probability :math:`2c/(d-1)`.

    Args:
        d: Number of nodes.
        c: Expected number of edges per node.
        shuffle (optional): Randomly permute the nodes (otherwise upper triangular).
        rng (optional): Seed or np.random.Generator.

    Returns:
        Binary adjacency matrix (:math:`d \times d` np.array, ``B[i, j] = 1`` means :math:`i \to j`).
    """
    assert d >= 2, 'd must be at least 2'
    p = 2 * c / (d - 1)
    assert 0 <= p <= 1, f'edge probability p = 2c/(d-1) = {p} must be in [0, 1]'
    rng = np.random.default_rng(rng)
    B = np.triu(rng.random((d, d)) < p, k=1).astype(float)
    return _permute_nodes(B, rng) if shuffle else B


def SF_DAG(d, c, shuffle=False, rng=None):
    r"""
    Scale-free DAG with :math:`c(d-c)` edges by Barabási–Albert preferential attachment, as in ``networkx.barabasi_albert_graph``. Edges point from newer to older nodes, so hubs have large in-degree.

    Args:
        d: Number of nodes.
        c: Number of edges per new node (integer, :math:`1 \leq c < d`).
        shuffle (optional): Randomly permute the nodes (otherwise upper triangular).
        rng (optional): Seed or np.random.Generator.

    Returns:
        Binary adjacency matrix (:math:`d \times d` np.array, ``B[i, j] = 1`` means :math:`i \to j`).
    """
    assert c == int(c) and 1 <= c < d, 'c must be an integer with 1 <= c < d'
    c = int(c)
    rng = np.random.default_rng(rng)
    B = np.zeros((d, d))
    # initial star: nodes 1, ..., c point to the hub 0
    B[1:c + 1, 0] = 1
    # every node appears once per incident edge, so that uniform draws
    # from this list are proportional to degree
    repeated = [0] * c + list(range(1, c + 1))
    for new in range(c + 1, d):
        targets = set()
        while len(targets) < c:
            targets.add(repeated[rng.integers(len(repeated))])
        targets = list(targets)
        B[new, targets] = 1
        repeated += targets + [new] * c
    # reverse node labels such that edges point from lower to higher index
    B = B[::-1, ::-1].copy()
    return _permute_nodes(B, rng) if shuffle else B


def sample_ER_summary_graph(p, edge_prob, rng=None, max_tries=10_000):
    r"""
    Summary graph for time series: an Erdős–Rényi digraph (each :math:`i \to j`, :math:`i \neq j`, independently with probability ``edge_prob``, so possibly cyclic) plus a self-loop at every node, conditioned on being strongly connected. Due to the conditioning, edges are no longer independent.

    Args:
        p: Number of variables.
        edge_prob: Probability of each edge.
        rng (optional): Seed or np.random.Generator.
        max_tries (optional): Maximum number of graphs to sample.

    Returns:
        Binary adjacency matrix of a directed, possibly cyclic graph (:math:`p \times p` np.array, ``B[i, j] = 1`` means :math:`i \to j`).
    """
    rng = np.random.default_rng(rng)
    for _ in range(max_tries):
        B = rng.binomial(1, edge_prob, size=(p, p))
        np.fill_diagonal(B, 1)
        if is_strongly_connected(B):
            return B
    raise RuntimeError(
        f'no strongly connected summary graph after {max_tries} tries, try increasing edge_prob'
    )


### -------------------------time-unrolled DAGs---------------------------------


def time_unrolled_DAG(A_summary, T):
    r"""
    Unroll a summary graph with lag-1 edges over :math:`T` time steps: ``A_summary[i, j] != 0`` means :math:`i(t) \to j(t+1)` for :math:`t = 0, \dots, T-2`. Node :math:`i` at time :math:`t` has index :math:`tp + i`. The Kronecker product with the :math:`T \times T` shift matrix used here is a shorthand for iterating over time steps and placing ``A_summary`` in each block :math:`(t, t+1)`.

    Args:
        A_summary: Weighted/Binary adjacency matrix of a directed graph, possibly cyclic and with self-loops (:math:`p \times p` np.array).
        T: Number of time steps.

    Returns:
        Adjacency matrix of the time-unrolled DAG (:math:`pT \times pT` np.array).
    """

    return np.kron(np.eye(T, k=1), A_summary)


### -------------------------graph properties-----------------------------------


def reachability(W):
    """ Boolean reachability matrix R of the directed graph W (possibly cyclic), R[i, j] = (i == j or there is a directed path i -> j). """
    R = (W != 0) | np.eye(len(W), dtype=bool)
    # R^(2^k) covers all paths of length <= 2^k, so stop once squaring changes nothing
    while True:
        R_f = R.astype(np.float32)
        R_next = (R_f @ R_f) > 0
        if np.array_equal(R_next, R):
            return R
        R = R_next


def is_strongly_connected(W):
    """ Whether every node of the directed graph W (possibly cyclic) can reach every other node. """
    return reachability(W).all()


def get_relatives(W):
    r"""
    Get the relatives of each node in a DAG, as a list of boolean masks.

    Args:
        W: Weighted/Binary DAG adjacency matrix (:math:`d \times d` np.array, ``W[i, j] != 0`` means :math:`i \to j`).
    """
    n = len(W)
    R = reachability(W)
    # relatives of i = ancestors of i union descendants of all ancestors of i
    relatives_list = []
    for i in range(n):
        anc = R[:, i]
        # take union of anc and all nodes that are descendants of any anc
        relatives_list.append(anc | R[anc].any(axis=0))
    return relatives_list


def count_relatives(W):
    r"""
    Count the relatives of each node in a DAG.
    Idea: for every node, get set of ancestors and set of descendants; then take appropriate unions to get the relatives.

    Args:
        W: Weighted/Binary DAG adjacency matrix (:math:`d \times d` np.array, ``W[i, j] != 0`` means :math:`i \to j`).
    """
    relatives_list = get_relatives(W)
    n_relatives = np.array([relatives.sum() for relatives in relatives_list])
    return n_relatives
