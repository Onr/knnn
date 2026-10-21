"""Compare scalar and batched kernels; run as python -m scripts.benchmark_batches."""
import gc
import json
import time
import tracemalloc

import numpy as np

from knnn.knnn_main import EPSILON, calc_eigen_for_set


def measure(function):
    function()  # Warm up the numerical libraries.
    durations = []
    for _ in range(3):
        gc.collect()
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'median_seconds': float(np.median(durations)),
            'peak_mib': peak / 1024 ** 2}


def main():
    rng = np.random.default_rng(42)
    n_train, n_test, dimensions, k_eigen, k_query = 1500, 2000, 25, 50, 3
    train = rng.normal(size=(n_train, dimensions))
    indices = rng.integers(n_train, size=(n_train, k_eigen))

    def scalar_fit():
        values, vectors = [], []
        for neighborhood in indices:
            value, vector = np.linalg.eigh(np.cov(train[neighborhood], rowvar=False))
            values.append(np.sqrt(np.maximum(value, 0)))
            vectors.append(vector)
        return np.array(values), np.array(vectors)

    def batch_fit():
        return calc_eigen_for_set(train, indices, 256)

    queries = rng.normal(size=(n_test, dimensions))
    query_indices = rng.integers(n_train, size=(n_test, k_query))
    values, vectors = batch_fit()

    def project(batch_size):
        scores = np.empty(n_test)
        for start in range(0, n_test, batch_size):
            stop = min(n_test, start + batch_size)
            current = query_indices[start:stop]
            differences = queries[start:stop, None] - train[current]
            projections = np.einsum('nkd,nkdl->nkl', differences,
                                    vectors[current], optimize=True)
            scores[start:stop] = (np.abs(projections)
                                 / (values[current] + EPSILON)).mean(axis=(1, 2))
        return scores

    report = {
        'shape': {'train': n_train, 'test': n_test, 'dimensions': dimensions,
                  'k_eigen': k_eigen, 'k_query': k_query},
        'scalar_eigendecomposition': measure(scalar_fit),
        'batch_eigendecomposition': measure(batch_fit),
        'unbatched_projection': measure(lambda: project(n_test)),
        'batched_projection': measure(lambda: project(256)),
        'max_eigenvalue_delta': float(np.max(np.abs(scalar_fit()[0] - batch_fit()[0]))),
        'max_score_delta': float(np.max(np.abs(project(n_test) - project(256)))),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
