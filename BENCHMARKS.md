# Validation of 0.1.0

The audit referenced v0.1.5, but the repository and PyPI were both at v0.0.9.
Its benchmark data and generated report were not supplied. The results below
are independent measurements; they do not reproduce the audit's experiment.

## Numerical and API checks

The regression suite checks the L1 formula against scalar sample-covariance
calculations, L2 against direct Mahalanobis distances, singular covariance
fallback, correlated feature grouping, one-dimensional remainders, constant
features, batch-size invariance, bounded covariance/projection batches,
constructor preservation, and classifier labels. Both KNNN and KNNNClassifier
pass sklearn's standard estimator checks. KNNN retains continuous `predict`
scores, so it intentionally does not inherit OutlierMixin.

All 57 correctness/API tests passed on Python 3.12 with NumPy 2.5.4 and
sklearn 1.9.1, and on Python 3.10 with NumPy 1.21.6 and sklearn 1.2.2.
The latter run used the installed wheel. A separate clean wheel installation
also passed runtime, cloning, serialization and classifier pipeline checks
without FAISS, tqdm, matplotlib or pytest installed. Wheel and source archive
metadata passed `twine check`. Optional array API checks were skipped by
sklearn; the minimal Python 3.10 environment also skipped pandas-specific checks.

## Batch kernels

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.benchmark_batches
```

One local run used Python 3.12, NumPy 2.5.4, 1,500 training samples, 2,000
queries, 25 dimensions, 50 covariance neighbors, 3 query neighbors, and
batches of 256. Times are medians of three runs after a warmup. Memory is peak
Python/NumPy allocation tracked by tracemalloc, not total process RAM.

| Calculation | Original-style kernel | Batched kernel |
| --- | ---: | ---: |
| Eigendecomposition | 387.2 ms | 194.7 ms |
| L1 projections | 11.80 ms | 9.49 ms |
| Projection peak allocation | 30.92 MiB | 4.12 MiB |
| Eigen fit peak allocation | 15.31 MiB | 17.36 MiB |

Eigendecomposition was about 1.99x faster, projection about 1.24x faster, and
projection peak allocation about 7.50x lower. Maximum standard-deviation
(eigenvalue square root) difference was 4.7e-15; projection score difference
was zero. Batched fitting uses extra covariance workspace, as the last row
shows. These are kernel measurements, not end-to-end fit/predict timings.

## Existing comparisons

Run the existing statistical comparisons with:

```bash
python3 -m pytest -m "benchmark and not external"
```

Using the default L2 scoring and the existing splits/seeds:

| Dataset | k-NNN AUROC | k-NN AUROC |
| --- | ---: | ---: |
| Moons | 0.9008 | 0.6761 |
| Swiss roll | 0.8595 | 0.5362 |
| Circles | 0.8860 | 0.4924 |
| Blobs | 0.9170 | 0.8769 |
| Moons + blobs | 0.8961 | 0.7356 |
| Breast cancer | 0.9675 | 0.8494 |
| Iris | 0.9854 | 0.9854 |
| Digits | 0.9832 | 0.9922 |
| Wine | 0.9905 | 0.7336 |

Seven comparisons pass. Two retain failing strict-superiority assertions:
Iris is a tie (also a failure on unchanged v0.0.9); Digits favors k-NN.
These failures are reported, not skipped or weakened. On the unchanged
v0.0.9 Digits split, k-NNN scored 0.9872; the algorithms' feature grouping,
retained dimensions and neighbor-distance behavior differ between versions.

Strict paper-style L1 weighting (`p=1`) scored 0.9452 on Digits, which is why
L2 remains the default. Changing defaults is not justified by the audit's
claimed double-root bug: that code does not exist in this checkout.

The breast-cancer comparison's k-NN score sign was corrected to match its
normal-class labels and the k-NNN score sign. No algorithm's universal
superiority is asserted by the correctness suite.

The 195 ADB comparisons now skip explicitly when their local data is absent;
three Hugging Face comparisons need downloaded datasets and were not run.
