import logging

import numpy as np

from knnn import KNNN, KNNN_class, create_samples, draw_to_points, synthetic_dataset_types


def test_knnn_default_config():
    knnn = KNNN()
    knnn.fit(np.random.rand(100, 10))
    res = knnn(np.random.rand(7, 10))
    assert res is not None, "KNNN returned None"


def test_knnn_classification_default_config():
    num_of_classes = 4
    knnn_class = KNNN_class(number_of_classes=num_of_classes)
    knnn_class.fit(np.random.rand(100, 11), gt=np.random.randint(0, num_of_classes, 100))
    res = knnn_class(np.random.rand(50, 11))
    assert res is not None, "KNNN returned None"

def test_knnn_return_shape():
    num_of_test_samples = 6
    number_of_neighbors = 7
    knnn_configs = {
        "number_of_neighbors": number_of_neighbors,
        "number_of_neighbors_of_neighbors": 15,
    }
    knnn = KNNN(**knnn_configs)
    knnn.fit(np.random.rand(100, 11))
    res_knnn, res_knn_all = knnn(np.random.rand(num_of_test_samples, 11), return_nearest_neigbours_results=True)
    assert res_knnn.shape == (num_of_test_samples,), "KNNN returned wrong shape"
    assert res_knn_all['knn_indeises'].shape == (num_of_test_samples,number_of_neighbors), "KNNN returned wrong shape"
    assert res_knn_all['knn_distance'].shape == (num_of_test_samples,number_of_neighbors), "KNNN returned wrong shape"


def test_knnn_classification_return_shape():
    num_of_test_samples = 6
    number_of_neighbors = 7
    num_of_classes = 3
    knnn_configs = {
        "number_of_neighbors": number_of_neighbors,
        "number_of_neighbors_of_neighbors": 15,
    }
    knnn_class = KNNN_class(number_of_classes=num_of_classes, **knnn_configs)
    knnn_class.fit(np.random.rand(100, 11), gt=np.random.randint(0, num_of_classes, 100))
    res_knnn, res_knn, knnn_results_by_class_bin, knn_results_by_class_bin = knnn_class(np.random.rand(num_of_test_samples, 11), return_nearest_neigbours_results=True)

    assert knnn_results_by_class_bin.shape == (
        num_of_test_samples, num_of_classes), "KNNN returned wrong shape"
    assert knn_results_by_class_bin.shape == (
        num_of_test_samples, num_of_classes), "KNNN returned wrong shape"
    assert res_knn.shape == (
        num_of_test_samples, num_of_classes), "KNN returned wrong shape"
    assert res_knnn.shape == (
        num_of_test_samples, num_of_classes), "KNNN returned wrong shape"


