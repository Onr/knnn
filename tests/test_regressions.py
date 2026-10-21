"""Numerical and API regressions for the algorithm and batched implementation."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knnn import KNNN, KNNN_class, KNNNClassifier
from knnn.knnn_main import (
    EPSILON, calc_eigen_for_set, compute_correlation_matrix,
    reorder_features_by_correlation,
)


def test_correlated_pairs_are_in_the_same_blocks():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(2, 200))
    data = np.column_stack([a, b, -a, b * 2, np.ones(len(a))])
    order = reorder_features_by_correlation(data, L=2)
    assert_array_equal(order[:4], [0, 2, 1, 3])
    model = KNNN(set_size=2).fit(data)
    assert model.features_clusters[0] == model.features_clusters[2]
    assert model.features_clusters[1] == model.features_clusters[3]
    assert np.isfinite(compute_correlation_matrix(data)).all()
    assert_array_equal(np.sort(order), np.arange(data.shape[1]))


def test_reordering_follows_previous_two_features_and_remaining_indices():
    data = np.random.default_rng(11).normal(size=(80, 13))
    correlation = np.abs(np.corrcoef(data, rowvar=False))
    order = list(range(data.shape[1]))
    L = 4
    for i in range(1, len(order)):
        previous = order[max(i - 2, (i // L) * L) : i] if i % L else order[max(0, i - 2) : i]
        scores = [np.mean([correlation[p, candidate] for p in previous])
                  for candidate in order[i:]]
        selected = i + (np.argmin(scores) if i % L == 0 else np.argmax(scores))
        order[i], order[selected] = order[selected], order[i]
    assert_array_equal(reorder_features_by_correlation(data, L), order)


@pytest.mark.parametrize('dimensions', [1, 2, 4, 5, 6, 26, 51, 76, 101])
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_all_dimensions_are_used_and_scores_are_finite(dimensions, metric):
    rng = np.random.default_rng(20)
    data = rng.normal(size=(35, dimensions))
    model = KNNN(set_size=25, distance_function=metric, batch_size=7).fit(data)
    used = np.concatenate(model.feature_indices_)
    assert_array_equal(np.sort(used), np.arange(dimensions))
    assert np.isfinite(model.predict(rng.normal(size=(9, dimensions)))).all()


@pytest.mark.parametrize('dimensions', [1, 4])
def test_covariance_batch_matches_scalar_sample_covariance(dimensions):
    data = np.random.default_rng(17).normal(size=(35, dimensions))
    indices = np.array([np.roll(np.arange(35), i)[:12] for i in range(35)])
    values, vectors = calc_eigen_for_set(data, indices, batch_size=6)
    for i in range(len(data)):
        covariance = np.atleast_2d(np.cov(data[indices[i]], rowvar=False))
        expected, _ = np.linalg.eigh(covariance)
        assert_allclose(values[i] ** 2, expected, atol=1e-14)
        assert_allclose(vectors[i] @ np.diag(values[i] ** 2) @ vectors[i].T,
                        covariance, atol=1e-14)
    if dimensions == 1:
        assert_allclose(values[:, 0] ** 2, np.var(data[indices, 0], axis=1, ddof=1))


def test_scores_match_paper_formula_and_dimension_weighting():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(45, 5)) * [1, 2, 4, 8, 16]
    queries = rng.normal(size=(9, 5))
    model = KNNN(number_of_neighbors=3, number_of_neighbors_of_neighbors=14,
                 set_size=2, p=1, distance_function='euclidean').fit(data)
    actual, diagnostics = model(queries, True)
    expected = []
    for query, neighbors in zip(queries, diagnostics['knn_indeises']):
        total = 0
        for neighbor in neighbors:
            local = model.neighbors_.kneighbors(data[neighbor:neighbor + 1],
                                                n_neighbors=14, return_distance=False)[0]
            for features in model.feature_indices_:
                covariance = np.atleast_2d(np.cov(data[local][:, features], rowvar=False))
                values, vectors = np.linalg.eigh(covariance)
                total += np.sum(np.abs((query[features] - data[neighbor, features]) @ vectors)
                                / (np.sqrt(np.maximum(values, 0)) + EPSILON))
        expected.append(total / (len(neighbors) * data.shape[1]))
    assert_allclose(actual, expected, rtol=1e-12)
    assert_allclose(diagnostics['knn_distance'],
                    np.sum((queries[:, None] - data[diagnostics['knn_indeises']]) ** 2, axis=2))


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_batch_size_does_not_change_scores_or_diagnostics(p, metric):
    rng = np.random.default_rng(9)
    data, queries = rng.normal(size=(50, 7)), rng.normal(size=(19, 7))
    small = KNNN(set_size=3, batch_size=4, p=p, distance_function=metric).fit(data)
    large = KNNN(set_size=3, batch_size=100, p=p, distance_function=metric).fit(data)
    a, da = small(queries, True)
    b, db = large(queries, True)
    assert_allclose(a, b)
    assert_array_equal(da['knn_indeises'], db['knn_indeises'])
    assert_allclose(da['knn_distance'], db['knn_distance'])


def test_eigendecomposition_is_bounded_by_batch_size(monkeypatch):
    original = np.linalg.eigh
    sizes = []
    def record(covariance):
        sizes.append(covariance.shape[0])
        return original(covariance)
    monkeypatch.setattr(np.linalg, 'eigh', record)
    KNNN(batch_size=6).fit(np.random.default_rng(0).normal(size=(33, 11)))
    assert max(sizes) <= 6
    assert len(sizes) > 1


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_constant_and_duplicate_data_are_supported(metric):
    data = np.ones((15, 6))
    model = KNNN(set_size=5, distance_function=metric).fit(data)
    assert_allclose(model(data), 0)
    assert model(np.zeros((1, 6)))[0] == np.finfo(float).max
    assert np.isfinite(model(data)).all()


def test_trailing_feature_anomaly_is_detected():
    data = np.random.default_rng(0).normal(size=(80, 6))
    model = KNNN(set_size=5, distance_function='euclidean').fit(data)
    queries = data[:5].copy()
    scores = model(queries)
    queries[:, -1] += 100
    assert np.all(model(queries) > scores)


def test_small_data_clamps_effective_counts_without_changing_parameters():
    data = np.array([[0.], [1.], [3.]])
    model = KNNN(number_of_neighbors=10, number_of_neighbors_of_neighbors=50, set_size=-1)
    before = model.get_params()
    assert model.fit(data) is model
    assert model.get_params() == before
    assert model.n_neighbors_ == model.n_eigen_neighbors_ == 3
    assert model.set_size_ == 1
    assert_allclose(clone(model).fit(data)(data), model(data))
    assert_allclose(model.score_samples(data), -model.predict(data))
    previous = model(data)
    data[:] = 100
    assert_allclose(model(np.array([[0.], [1.], [3.]])), previous)


@pytest.mark.parametrize('parameter,value', [
    ('batch_size', 0), ('batch_size', True), ('set_size', 0), ('set_size', -2),
    ('number_of_neighbors', 0), ('number_of_neighbors_of_neighbors', 1),
    ('distance_function', 'invalid'), ('p', 3),
])
def test_invalid_parameters_raise_clear_errors(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        KNNN(**{parameter: value}).fit(np.ones((5, 3)))


def test_input_validation_and_unfitted_errors():
    with pytest.raises(NotFittedError):
        KNNN().predict([[1., 2.]])
    with pytest.raises(NotFittedError):
        KNNNClassifier().predict([[1., 2.]])
    with pytest.raises(ValueError):
        KNNN().fit([[1., 2.]])
    model = KNNN().fit(np.ones((4, 3)))
    with pytest.raises(ValueError, match='features'):
        model.predict([[1., 2.]])
    with pytest.raises(ValueError):
        model.predict([[np.nan, 0, 0]])


def test_classifier_pipeline_grid_search_and_label_order():
    rng = np.random.default_rng(42)
    data = np.r_[rng.normal(-5, .1, size=(24, 3)), rng.normal(5, .1, size=(24, 3))]
    labels = np.array(['zebra'] * 24 + ['apple'] * 24)
    model = KNNNClassifier(distance_function='euclidean', set_size=2)
    pipeline = make_pipeline(StandardScaler(), model)
    search = GridSearchCV(pipeline, {'knnnclassifier__number_of_neighbors': [1, 3]}, cv=3)
    search.fit(data, labels.tolist())
    assert search.score(data, labels) == 1
    assert is_classifier(model)
    fitted = search.best_estimator_[-1]
    assert_array_equal(fitted.classes_, ['apple', 'zebra'])
    assert_allclose(search.predict_proba(data).sum(axis=1), 1)
    assert clone(fitted).get_params() == fitted.get_params()


def test_anomaly_estimator_pipeline_and_grid_search():
    rng = np.random.default_rng(2)
    data = np.r_[rng.normal(size=(30, 2)), rng.normal(8, 1, size=(30, 2))]
    labels = np.r_[np.zeros(30), np.ones(30)]
    def scorer(estimator, X, y):
        return roc_auc_score(y, estimator.predict(X))
    search = GridSearchCV(make_pipeline(StandardScaler(), KNNN()),
                          {'knnn__set_size': [1, 2]}, scoring=scorer,
                          cv=StratifiedKFold(3))
    search.fit(data, labels)
    assert np.isfinite(search.best_score_)


def test_legacy_classifier_cloning_refitting_and_tuples():
    data = np.array([[0., 0.], [1., 0.], [0., 1.], [10., 10.], [11., 10.], [10., 11.]])
    labels = ['b'] * 3 + ['a'] * 3
    model = KNNN_class(number_of_classes=2, distance_function='euclidean')
    before = model.get_params()
    assert model.fit(data, gt=labels) is model
    assert model.get_params() == before
    assert len(model(data)) == 4
    assert len(model(data, False)) == 2
    assert_allclose(model(data)[0], clone(model).fit(data, y=labels)(data)[0])
    assert_array_equal(model.predict(data), labels)
    model.fit(data, gt=['d'] * 3 + ['c'] * 3)
    assert_array_equal(model.classes_, ['c', 'd'])
    assert len(model.knnn_by_class) == 2


def test_default_l2_matches_mahalanobis_with_sample_covariance():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(30, 4)) * [1, 2, 3, 5]
    queries = rng.normal(size=(7, 4))
    model = KNNN(set_size=-1, number_of_neighbors_of_neighbors=len(data),
                 distance_function='euclidean').fit(data)
    scores, diagnostics = model(queries, True)
    inverse = np.linalg.inv(np.cov(data, rowvar=False))
    expected = []
    for query, neighbors in zip(queries, diagnostics['knn_indeises']):
        differences = query - data[neighbors]
        distances = [np.sqrt(difference @ inverse @ difference / data.shape[1])
                     for difference in differences]
        expected.append(np.mean(distances))
    assert_allclose(scores, expected, rtol=1e-8)


def test_default_l2_uses_euclidean_fallback_for_singular_covariance():
    line = np.linspace(-2, 2, 30)
    data = np.column_stack([line, line])
    queries = np.array([[0, 1.], [1., -1.]])
    model = KNNN(set_size=2, distance_function='euclidean').fit(data)
    scores, diagnostics = model(queries, True)
    differences = queries[:, None] - data[diagnostics['knn_indeises']]
    assert_allclose(scores, np.sqrt(np.mean(differences ** 2, axis=2)).mean(axis=1))
    assert np.isfinite(scores).all()


def test_projection_arrays_are_bounded_by_batch_size(monkeypatch):
    rng = np.random.default_rng(0)
    model = KNNN(batch_size=4).fit(rng.normal(size=(35, 11)))
    original = np.einsum
    sizes = []
    def record(subscripts, *operands, **kwargs):
        if subscripts == 'nkd,nkdl->nkl':
            sizes.append(operands[1].shape[0])
        return original(subscripts, *operands, **kwargs)
    monkeypatch.setattr(np, 'einsum', record)
    model.predict(rng.normal(size=(19, 11)))
    assert max(sizes) <= 4
    assert len(sizes) > 1


@pytest.mark.parametrize('estimator', [KNNN(), KNNNClassifier()])
def test_sklearn_estimator_checks(estimator):
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(estimator)
