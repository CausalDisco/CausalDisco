# Changelog

## 0.3.0

### Breaking changes
- Sortability functions (`var_sortability`, `r2_sortability`, `snr_sortability`, `rel_sortability`, `rel_sortability_oracle`) take `measure` in `"edges"` (new default), `"ancestors"` (new), or `"path_lengths"`. The previous default `"paths"` is now called `"path_lengths"` and remains available to reproduce the values of Reisach et al. (2021, 2023); default values change accordingly. Unknown measures raise a `ValueError`.
- `relatives_sortability` is renamed to `rel_sortability`, and `true_relatives_sortability` to `rel_sortability_oracle`.
- `order_alignment_adjacent_pairs` is renamed to `order_alignment_edges`, and `order_alignment_paths` to `order_alignment_path_lengths`.
- `sort_regress` (and thus `random_sort_regress`, `var_sort_regress`, `r2_sort_regress`) returns OLS refits of each variable onto the parents selected by the adaptive Lasso, instead of the shrunk adaptive Lasso coefficients. The selected edges are unchanged.
- `reachability`, `get_relatives`, and `count_relatives` move from `CausalDisco.analytics` to `CausalDisco.graphs`.
- `r2coeff` and `rel_count_emp_hard` (and thus the sortabilities and baselines based on them) raise a `ValueError` for constant variables, whose correlations are undefined. Previously, `r2coeff` raised an uninformative error and `rel_count_emp_hard` silently returned a count of 0.
- Requires Python >= 3.10.

### New
- `order_alignment_ancestors`: order alignment over all ancestor-descendant pairs.
- `rel_sort_regress`: sort-and-regress baseline using the empirical number of relatives.
- `CausalDisco.graphs`: Erdős–Rényi (`ER_DAG`) and scale-free (`SF_DAG`) random DAG models.
- `CausalDisco.graphs`: random Erdős–Rényi summary graphs for time series (`sample_ER_summary_graph`), their unrolling into DAGs (`time_unrolled_DAG`), and `is_strongly_connected`.
- New dependency `gadjid` (graph distances such as the SHD, used in the examples).
- `sort_regress` has an optional `coef_tol` argument for pruning small adaptive Lasso coefficients.

### Fixes
- `order_alignment_edges` and `count_relatives` now handle weighted adjacency matrices correctly (every nonzero entry is treated as an edge). Results for binary adjacency matrices are unchanged.
- `sort_regress` no longer returns NaN for constant columns.
