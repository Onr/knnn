# Changelog

## 0.1.0

- Retain L2 Mahalanobis scoring as the default (`p=2`), using RMS aggregation
  and stable Euclidean fallback for singular local covariances. Add `p=1` for
  the paper's absolute projections divided by local covariance standard
  deviations. Existing score thresholds must be recalibrated because grouping,
  partial blocks and score normalization have changed.
- Keep paper-style L1 scoring opt-in: the alleged double-root bug is absent
  from v0.0.9, and applying strict weighting to singular sparse neighborhoods
  reduced accuracy on the Digits comparison.
- Greedily group correlated features using the actual `set_size`, averaging
  correlations with the preceding two entries as described in Section 3.
  Keep constant features at the end so they do not split correlated blocks.
- Retain every feature, including partial blocks and single-feature blocks.
  Combine block means in proportion to their feature counts.
- Batch covariance construction, eigendecomposition, neighbor queries and
  inference projections with configurable `batch_size` (default 256).
- Use scikit-learn nearest neighbors instead of requiring FAISS at runtime.
  The legacy FAISS helper is available with the `faiss` extra. Cosine distance
  now follows sklearn's convention for zero vectors rather than producing NaNs.
- Add estimator cloning, parameter search, fitting with an ignored `y`, returning
  `self`, fitted checks and `n_features_in_`. Effective neighbor counts are
  clamped without changing constructor parameters. `set_size=-1` uses all
  features. The legacy `predict` anomaly-score contract remains supported;
  `score_samples` returns negative anomaly scores.
- Add `KNNNClassifier` with sorted labels, class predictions, tie-split votes,
  Pipeline and GridSearchCV support. Retain `KNNN_class` and its tuple results.
- Move packaging metadata to pyproject.toml; provide working dev and tests
  extras, require Python >=3.9, and exclude test modules from the wheel.
- Correct the clone command and proceedings name. The official citation is
  WACV Workshops 2024, pp. 1005-1014, not the pages proposed in the audit.
- Add numerical and API regressions. Mark comparison and external-data
  benchmarks explicitly; skip unavailable ADB datasets instead of passing
  without evaluating any data.

The supplied audit referred to v0.1.5 and functions absent from the checked-out
v0.0.9 code (also the latest PyPI release when reviewed). These changes adapt
its useful recommendations to this implementation. Its reported AUROC,
speed and memory improvements have not been independently reproduced.
