import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score #balanced_accuracy_score, precision_score, recall_score, f1_score 
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from datasets import load_dataset

from knnn import KNNN, KNNN_class
from tests.utils import document_results, plot_and_save_synthetic

SEED = 42

# sklearn dataset

def test_knn_vs_knnn_on_brest_cancer_data():
    dataset = load_breast_cancer()
    test_sample_num = 150
    np.random.seed(SEED)
    random_idx_s = np.random.choice(range(len(dataset['data'])), len(dataset['data']), replace=False)
    data_train = dataset['data'][random_idx_s[:-test_sample_num]]
    gt_train_s = dataset['target'][random_idx_s[:-test_sample_num]]
    data_test = dataset['data'][random_idx_s[-test_sample_num:]]
    gt_test_s = dataset['target'][random_idx_s[-test_sample_num:]]
    # remove anomalies from the train set
    data_train = data_train[gt_train_s == 1]
    knnn_configs = {
        # "number_of_neighbors": 100, # 3
        # "number_of_neighbors_of_neighbors": 25,
        # "set_size": 4,  # 2
        # "neighbors_agg_func": "min3mean",
    }
    knnn = KNNN(**knnn_configs)
    knnn.fit(data_train)
    res, res_knn = knnn(data_test, return_nearest_neigbours_results=True)
    knn_res = res_knn['knn_distance'].mean(1)
    knnn_res = -res
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, knn_res)
    roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_res)
    average_precision_score_knn = average_precision_score(gt_test_s, knn_res)
    average_precision_score_knnn = average_precision_score(gt_test_s, knnn_res)

    # document results
    results_dict = {
        "dataset": 'brest_cancer',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn.number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
        "set_size": knnn.set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_is_better": roc_auc_score_knnn > roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
        "knnn_ap_is_better": average_precision_score_knnn > average_precision_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > 0.6, f"something is wrong with the knnn on brest cancer dataset (knnn={roc_auc_score_knnn})"
    assert roc_auc_score_knn > 0.6, f"something is wrong with the knn on brest cancer dataset (knn={roc_auc_score_knn})"
    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knn > 0.6, f"something is wrong with the knnn on brest cancer dataset (knnn={average_precision_score_knn})"
    assert average_precision_score_knnn > 0.6, f"something is wrong with the knnn on brest cancer dataset (knnn={average_precision_score_knnn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"
 

def test_knn_vs_knnn_on_iris_data():
    dataset = load_iris(as_frame=True)
    number_of_classes = len(dataset['target'].unique())
    test_sample_num = 50
    np.random.seed(SEED)
    random_idx_s = np.random.choice(range(len(dataset['data'])), len(dataset['data']), replace=False)
    data_train = dataset['data'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    gt_train_s = dataset['target'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    data_test = dataset['data'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    gt_test_s = dataset['target'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    knnn_configs = {
        "number_of_neighbors": 5, 
        "number_of_neighbors_of_neighbors": 25, 
        "set_size": -1,  
    }
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    _, _, knnn_results_by_class_bin, knn_results_by_class_bin = knnn_class(data_test, return_nearest_neigbours_results=True)
    
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, knn_results_by_class_bin, multi_class='ovr')
    roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_results_by_class_bin, multi_class='ovr')

    # document results
    results_dict = {
        "dataset": 'iris',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size":  knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
    }
    document_results(results_dict)
    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on iris dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
 

def test_knn_vs_knnn_on_digits_data():
    dataset = load_digits(as_frame=True)
    number_of_classes = len(dataset['target'].unique())
    test_sample_num = 300
    np.random.seed(SEED)
    random_idx_s = np.random.choice(range(len(dataset['data'])), len(dataset['data']), replace=False)
    data_train = dataset['data'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    gt_train_s = dataset['target'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    data_test = dataset['data'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    gt_test_s = dataset['target'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    knnn_configs = {
        # # "number_of_neighbors": 3, # 3
        # "number_of_neighbors_of_neighbors": 2, # 15
        # "set_size": 2,  # 2
        "distance_function": 'cosine', # 'cosine', 'euclidean'
    }
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    _, _, knnn_results_by_class_bin, knn_results_by_class_bin  = knnn_class(data_test, return_nearest_neigbours_results=True)
    
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, knn_results_by_class_bin, multi_class='ovr')
    roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_results_by_class_bin, multi_class='ovr')

    # document results
    results_dict = {
        "dataset": 'digits',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size":  knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on digits dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
 

def test_knn_vs_knnn_on_wine_data():
    dataset = load_wine(as_frame=True)
    number_of_classes = len(dataset['target'].unique())
    test_sample_num = 70
    np.random.seed(SEED)
    random_idx_s = np.random.choice(range(len(dataset['data'])), len(dataset['data']), replace=False)
    data_train = dataset['data'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    gt_train_s = dataset['target'].iloc[random_idx_s[:-test_sample_num]].to_numpy()
    data_test = dataset['data'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    gt_test_s = dataset['target'].iloc[random_idx_s[-test_sample_num:]].to_numpy()
    knnn_configs = {
        "number_of_neighbors": 25, # 3
        "number_of_neighbors_of_neighbors": 25, # 15
        "set_size": 3,  # 2
    }
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    _, _, knnn_results_by_class_bin, knn_results_by_class_bin = knnn_class(data_test, return_nearest_neigbours_results=True)
    
    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, knn_results_by_class_bin, multi_class='ovr')
    roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_results_by_class_bin, multi_class='ovr')

    # document results
    results_dict = {
        "dataset": 'wine',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size":  knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on wine dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
 

# huggingsface datasets

def test_knn_vs_knnn_on_inria_bioresponse(): # https://huggingface.co/datasets/inria-soda/tabular-benchmark
    test_sample_num = 500
    dataset_bio = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/Bioresponse.csv")['train'].to_pandas()
    np.random.seed(SEED)
    shuffled_indenes = np.random.permutation(dataset_bio.shape[0])
    dataset_train = dataset_bio.iloc[shuffled_indenes[:-test_sample_num]]
    dataset_test = dataset_bio.iloc[shuffled_indenes[-test_sample_num:]]
    # dataset_train = dataset_train[:1000] # TODO tmp remove this is for quicker debug
    gt_train_s = dataset_train['target'].to_numpy()
    gt_test_s = dataset_test['target'].to_numpy()
    number_of_classes = len(np.unique(gt_train_s))
    # drop target
    dataset_train = dataset_train.drop(columns=['target'])
    dataset_test = dataset_test.drop(columns=['target'])
    # convert to numpy
    data_train = dataset_train.to_numpy()
    data_test = dataset_test.to_numpy()  
    # run knnn and knn
    knnn_configs = {
        # "number_of_neighbors": 3, # 3
        # "number_of_neighbors_of_neighbors": 15,
        "set_size": 4,  # 2
    }
    # TODO tmp chnage
    # make sure that the number of features is divisible by the set size
    num_of_features_to_add = knnn_configs['set_size'] - data_train.shape[1] % knnn_configs['set_size']
    if num_of_features_to_add != 0:
        data_train = np.concatenate([data_train, 1e-5 * np.random.randn(data_train.shape[0], num_of_features_to_add)], axis=1)
        data_test = np.concatenate([data_test, 1e-5 * np.random.randn(data_test.shape[0], num_of_features_to_add)], axis=1)
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    knnn_res, knn_res, _, _ = knnn_class(data_test, return_nearest_neigbours_results=True)

    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, -knn_res[:, 1])
    roc_auc_score_knnn = roc_auc_score(gt_test_s, -knnn_res[:, 1])
    average_precision_score_knn = average_precision_score(gt_test_s, -knn_res[:, 1])
    average_precision_score_knnn = average_precision_score(gt_test_s, -knnn_res[:, 1])

    # document results
    results_dict = {
        "dataset": 'inria_bioresponse',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size": knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on bioresponse dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on bioresponse dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"
 
 
def test_knn_vs_knnn_on_inria_soda_eye(): # https://huggingface.co/datasets/inria-soda/tabular-benchmark # https://www.openml.org/search?type=data&sort=runs&id=1044
    # dataset_eye = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/eye_movements.csv")['train'].to_pandas()
    # remove irrelevant columns
    # dataset_eye = dataset_eye.drop(columns=['lineNo', 'assgNo', 'titleNo', 'wordNo'])
    # dataset_eye = dataset_eye.drop(columns=['P1stFixation', 'P2stFixation']) # todo test

    dataset_eye = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/eye_movements.csv")['train'].to_pandas()

    test_sample_num = 500
    np.random.seed(SEED)
    shuffled_indexes = np.random.permutation(dataset_eye.shape[0])
    dataset_train = dataset_eye.iloc[shuffled_indexes[:-test_sample_num]]
    dataset_test = dataset_eye.iloc[shuffled_indexes[-test_sample_num:]]

    # dataset_train = dataset_train[:1000] # TODO tmp remove this is for quicker debug
    gt_train_s = dataset_train['label'].to_numpy()
    gt_test_s = dataset_test['label'].to_numpy()
    number_of_classes = len(np.unique(gt_train_s))
    # drop label
    dataset_train = dataset_train.drop(columns=['label'])
    dataset_test = dataset_test.drop(columns=['label'])
    # convert to numpy
    data_train = dataset_train.to_numpy()
    data_test = dataset_test.to_numpy()  
    # run knnn and knn
    knnn_configs = {
        # "number_of_neighbors": 1,
        "number_of_neighbors_of_neighbors": 1500, 
        "set_size": -1,
    }
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    knnn_res, knn_res, _, _ = knnn_class(data_test, return_nearest_neigbours_results=True)

    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, -knn_res[:, 1])
    roc_auc_score_knnn = roc_auc_score(gt_test_s, -knnn_res[:, 1])
    average_precision_score_knn = average_precision_score(gt_test_s, -knn_res[:, 1])
    average_precision_score_knnn = average_precision_score(gt_test_s, -knnn_res[:, 1])
    logging.info(f"Results on eye movements dataset (AUROC: knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")

    # document results
    results_dict = {
        "dataset": 'inria_soda_eye',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size": knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on eye movements dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on eye movements dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"

    
def test_knn_vs_knnn_on_inria_soda_covertype(): # https://huggingface.co/datasets/inria-soda/tabular-benchmark 
    dataset_covertype = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/covertype.csv")['train'].to_pandas()
    # remove irrelevant columns
    dataset_covertype_filtered = dataset_covertype[dataset_covertype['Soil_Type33'] == 1]
    # remove all columns with Soil type in the name
    dataset_covertype_filtered = dataset_covertype_filtered.drop(columns=[col for col in dataset_covertype_filtered.columns if ('Soil_Type' in col) or  ('Wilderness_Area' in col)])

    test_sample_num = 2_000
    train_sample_num = 10_000 # -1
    np.random.seed(SEED)
    shuffled_indexes = np.random.permutation(dataset_covertype_filtered.shape[0])
    dataset_train = dataset_covertype_filtered.iloc[shuffled_indexes[:-test_sample_num]]
    dataset_train = dataset_train.iloc[:train_sample_num]
    dataset_test = dataset_covertype_filtered.iloc[shuffled_indexes[-test_sample_num:]]

    gt_train_s = dataset_train['class'].to_numpy() - 1 # class 1 is 0 in python
    gt_test_s = dataset_test['class'].to_numpy() - 1 # class 1 is 0 in python
    number_of_classes = len(np.unique(gt_train_s))
    # drop label
    dataset_train = dataset_train.drop(columns=['class'])
    dataset_test = dataset_test.drop(columns=['class'])
    # convert to numpy
    data_train = dataset_train.to_numpy()
    data_test = dataset_test.to_numpy()  


    # run knnn and knn
    knnn_configs = {
        # "set_size": 3,
    }
    knnn_class = KNNN_class(number_of_classes=number_of_classes ,**knnn_configs)
    knnn_class.fit(embedding=data_train, gt=gt_train_s)
    knnn_res, knn_res, _, _ = knnn_class(data_test)

    # compere scores
    roc_auc_score_knn = roc_auc_score(gt_test_s, -knn_res[:, 1])
    roc_auc_score_knnn = roc_auc_score(gt_test_s, -knnn_res[:, 1])
    average_precision_score_knn = average_precision_score(gt_test_s, -knn_res[:, 1])
    average_precision_score_knnn = average_precision_score(gt_test_s, -knnn_res[:, 1])
    logging.info(f"Results on covertypes dataset (AUROC: knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")
    
    # document results
    results_dict = {
        "dataset": 'inria_soda_covertype',
        "test_sample_num": test_sample_num,
        "train_sample_num": len(data_train),
        "number_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors,
        "number_of_neighbors_of_neighbors": knnn_class.knnn_by_class[0].number_of_neighbors_of_neighbors,
        "set_size": knnn_class.knnn_by_class[0].set_size,
        "seed": SEED,
        "knnn": roc_auc_score_knnn,
        "knn": roc_auc_score_knn,
        "knnn_ap": average_precision_score_knnn,
        "knn_ap": average_precision_score_knn,
    }
    document_results(results_dict)

    assert roc_auc_score_knnn > roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on covertypes dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
    assert average_precision_score_knnn > average_precision_score_knn, f"KNNN is not better than KNN by AP metric on covertypes dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"


if __name__ == "__main__":
    # test_knn_vs_knnn_on_brest_cancer_data()
    test_knn_vs_knnn_on_inria_bioresponse()
    # test_knn_vs_knnn_on_inria_soda_eye()
    # test_knn_vs_knnn_on_inria_soda_covertype()

