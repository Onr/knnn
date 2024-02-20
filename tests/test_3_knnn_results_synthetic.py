import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score #balanced_accuracy_score, precision_score, recall_score, f1_score 
from sklearn.datasets import load_breast_cancer
from datasets import load_dataset

from knnn import KNNN, KNNN_class, create_samples
from tests.utils import document_results, plot_and_save_synthetic

SEED = 42

def test_knnn_vs_knnn_on_synthetic_data_moons():
    np.random.seed(SEED)
    data_types = ['moons']
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=data_types)
    knnn_configs = {
        "number_of_neighbors":2, # 100, # 2
        # "number_of_neighbors_of_neighbors": 25,
        "set_size": -1,
    }
    knnn = KNNN(**knnn_configs)
    knnn.fit(xn_train_s)
    res_n_knnn, res_n_knn = knnn(xn_test_s, return_nearest_neigbours_results=True)
    res_a_knnn, res_a_knn = knnn(xa_test_s, return_nearest_neigbours_results=True)
    res_n_knn = res_n_knn['knn_distance'].mean(1)
    res_a_knn = res_a_knn['knn_distance'].mean(1)
    gt_s = np.concatenate((np.ones(xn_test_s.shape[0]), np.zeros(xa_test_s.shape[0])))
    knn_res = np.concatenate((res_n_knn, res_a_knn))
    knnn_res = np.concatenate((res_n_knnn, res_a_knnn))
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_s, -knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_s, -knnn_res)
    average_precision_score_knn = average_precision_score(gt_s, -knn_res)
    average_precision_score_knnn = average_precision_score(gt_s, -knnn_res)

    # document results
    results_dict = {
        "dataset": data_types,
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict)
  
    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"


def test_knnn_vs_knnn_on_synthetic_data_swiss_roll():
    np.random.seed(SEED)
    data_types = ['swiss_roll']
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=data_types)
    knnn_configs = {
        # "number_of_neighbors": 2,
        # "number_of_neighbors_of_neighbors": 25,
        "set_size": -1,
    }        
    knnn = KNNN(**knnn_configs)
    knnn.fit(xn_train_s)
    res_n_knnn, res_n_knn = knnn(xn_test_s, return_nearest_neigbours_results=True)
    res_a_knnn, res_a_knn = knnn(xa_test_s, return_nearest_neigbours_results=True)
    res_n_knn = res_n_knn['knn_distance'].mean(1)
    res_a_knn = res_a_knn['knn_distance'].mean(1)
    gt_s = np.concatenate((np.ones(xn_test_s.shape[0]), np.zeros(xa_test_s.shape[0])))
    knn_res = np.concatenate((res_n_knn, res_a_knn))
    knnn_res = np.concatenate((res_n_knnn, res_a_knnn))
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_s, -knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_s, -knnn_res)
    average_precision_score_knn = average_precision_score(gt_s, -knn_res)
    average_precision_score_knnn = average_precision_score(gt_s, -knnn_res)

    # document results
    results_dict = {
        "dataset": data_types,
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict) 

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"


def test_knnn_vs_knnn_on_synthetic_data_circles():
    np.random.seed(SEED+1)
    data_types = ['circles']
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=data_types)
    knnn_configs = {
        # "number_of_neighbors": 2,
        # "number_of_neighbors_of_neighbors": 25,
        "set_size": -1,
    }        
    knnn = KNNN(**knnn_configs)
    knnn.fit(xn_train_s)
    res_n_knnn, res_n_knn = knnn(xn_test_s, return_nearest_neigbours_results=True)
    res_a_knnn, res_a_knn = knnn(xa_test_s, return_nearest_neigbours_results=True)
    res_n_knn = res_n_knn['knn_distance'].mean(1)
    res_a_knn = res_a_knn['knn_distance'].mean(1)
    gt_s = np.concatenate((np.ones(xn_test_s.shape[0]), np.zeros(xa_test_s.shape[0])))
    knn_res = np.concatenate((res_n_knn, res_a_knn))
    knnn_res = np.concatenate((res_n_knnn, res_a_knnn))
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_s, -knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_s, -knnn_res)
    average_precision_score_knn = average_precision_score(gt_s, -knn_res)
    average_precision_score_knnn = average_precision_score(gt_s, -knnn_res)

    # document results
    results_dict = {
        "dataset": data_types,
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict) 
 
    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"


def test_knnn_vs_knnn_on_synthetic_data_blobs():
    np.random.seed(SEED)
    data_types = ['blobs']
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=data_types)
    knnn_configs = {
        # "number_of_neighbors": 100, #2  # TODO comment this line
        # "number_of_neighbors_of_neighbors": 25,
        "set_size": -1,
    }
    knnn = KNNN(**knnn_configs)
    knnn.fit(xn_train_s)
    res_n_knnn, res_n_knn = knnn(xn_test_s, return_nearest_neigbours_results=True)
    res_a_knnn, res_a_knn = knnn(xa_test_s, return_nearest_neigbours_results=True)
    res_n_knn = res_n_knn['knn_distance'].mean(1)
    res_a_knn = res_a_knn['knn_distance'].mean(1)
    gt_s = np.concatenate((np.ones(xn_test_s.shape[0]), np.zeros(xa_test_s.shape[0])))
    knn_res = np.concatenate((res_n_knn, res_a_knn))
    knnn_res = np.concatenate((res_n_knnn, res_a_knnn))
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_s, -knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_s, -knnn_res)
    average_precision_score_knn = average_precision_score(gt_s, -knn_res)
    average_precision_score_knnn = average_precision_score(gt_s, -knnn_res)

    # document results
    results_dict = {
        "dataset": data_types,
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict) 
 
    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"


def test_knnn_vs_knnn_on_synthetic_data_moons_blobs():
    np.random.seed(SEED)
    data_types = ['moons', 'blobs'] 
    xn_train_s, xn_test_s, xa_test_s = create_samples(normal_num_of_samples=100, data_types=data_types)

    # print('tmp')
    # from knnn import plot_and_save_synthetic
    # select_random_indies = np.random.choice(xn_test_s.shape[0], xn_train_s.shape[0], replace=False)
    # xn_test_s = xn_test_s[select_random_indies,:] 
    # xa_test_s = xa_test_s[select_random_indies,:]
    # plot_and_save_synthetic(train_n=xn_train_s, test_n=xn_test_s, test_a=xa_test_s, image_name="synthetic_dataset")
    # print('tmp')

    knnn_configs = {
        # "number_of_neighbors": 100, # 2
        # "number_of_neighbors_of_neighbors": 25,
        "set_size": -1,
    }
    knnn = KNNN(**knnn_configs)
    knnn.fit(xn_train_s)
    res_n_knnn, res_n_knn = knnn(xn_test_s, return_nearest_neigbours_results=True)
    res_a_knnn, res_a_knn = knnn(xa_test_s, return_nearest_neigbours_results=True)
    res_n_knn = res_n_knn['knn_distance'].mean(1)
    res_a_knn = res_a_knn['knn_distance'].mean(1)
    gt_s = np.concatenate((np.ones(xn_test_s.shape[0]), np.zeros(xa_test_s.shape[0])))
    knn_res = np.concatenate((res_n_knn, res_a_knn))
    knnn_res = np.concatenate((res_n_knnn, res_a_knnn))
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_s, -knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_s, -knnn_res)
    average_precision_score_knn = average_precision_score(gt_s, -knn_res)
    average_precision_score_knnn = average_precision_score(gt_s, -knnn_res)

    # document results
    results_dict = {
        "dataset": data_types,
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict) 

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"
  

if __name__ == "__main__":
    test_knnn_vs_knnn_on_synthetic_data_moons()
    test_knnn_vs_knnn_on_synthetic_data_swiss_roll()
    test_knnn_vs_knnn_on_synthetic_data_circles()
    test_knnn_vs_knnn_on_synthetic_data_blobs()
    test_knnn_vs_knnn_on_synthetic_data_moons_blobs()

