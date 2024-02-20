import logging

import numpy as np

from knnn import KNNN, create_samples, draw_to_points, synthetic_dataset_types


def test_create_dataset_default():
    nornal_num_of_samples = 100
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=nornal_num_of_samples)
    assert xn_train_s.shape == (nornal_num_of_samples, 2), f"create_samples returned wrong shape {xn_train_s.shape} != {(nornal_num_of_samples, 2)}"
    assert xn_test_s.shape == xa_test_s.shape, f"create_samples returned wrong shape {xn_test_s.shape} != {xa_test_s.shape}"
    assert len(np.unique(xn_test_s, axis=0)) == xn_test_s.shape[0], "create_samples returned not unique rows"
    assert len(np.unique(xa_test_s, axis=0)) == xa_test_s.shape[0], "create_samples returned not unique rows"
    assert len(np.unique(xn_train_s, axis=0)) == xn_train_s.shape[0], "create_samples returned not unique rows"
    assert xn_test_s.shape[0] == xa_test_s.shape[0], "create_samples returned different length of test normal and abnormal"


def test_create_dataset_different_seed():
    nornal_num_of_samples = 100
    seed = np.random.randint(10_000)
    xn_train_s_1, xn_test_s_1, xa_test_s_1 = create_samples(normal_num_of_samples=nornal_num_of_samples, seed=seed)
    xn_train_s_2, xn_test_s_2, xa_test_s_2 = create_samples(normal_num_of_samples=nornal_num_of_samples, seed=2*seed)
    assert not np.array_equal(xn_train_s_1, xn_train_s_2), "create_samples returned same data with different seeds"
    assert not np.array_equal(xn_test_s_1, xn_test_s_2), "create_samples returned same data with different seeds"
    assert not np.array_equal(xa_test_s_1, xa_test_s_2), "create_samples returned same data with different seeds"


def test_create_dataset_same_seed():
    nornal_num_of_samples = 100
    seed = np.random.randint(10_000)
    xn_train_s_1, xn_test_s_1, xa_test_s_1 = create_samples(normal_num_of_samples=nornal_num_of_samples, seed=seed)
    xn_train_s_2, xn_test_s_2, xa_test_s_2 = create_samples(normal_num_of_samples=nornal_num_of_samples, seed=seed)
    assert np.array_equal(xn_train_s_1, xn_train_s_2), "create_samples returned different data for the same seeds"
    assert np.array_equal(xn_test_s_1, xn_test_s_2), "create_samples returned different data for the same seeds"
    assert np.array_equal(xa_test_s_1, xa_test_s_2), "create_samples returned different data for the same seeds"


def test_create_dataset_type():
    for dataset_type in synthetic_dataset_types:
        xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=[dataset_type])
        assert xn_train_s.shape == (100, 2), f"create_samples returned wrong shape, dataset_type={dataset_type}"
        assert xn_test_s.shape == xa_test_s.shape, f"create_samples returned wrong shape, dataset_type={dataset_type}"
    

