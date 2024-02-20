import logging
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score #balanced_accuracy_score, precision_score, recall_score, f1_score 
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from datasets import load_dataset

from tests.utils import document_results, plot_and_save_synthetic
from knnn import KNNN, KNNN_class  


SEED = 42

MAIN_DATASET_PATH = Path('/home/onrh/projects/anomaly/new_patch_core/patchcore-inspection/test/datasets')
def run_test_on_ADB_dataset(dataset_type, dataset_name, knnn_configs=None): # TODO tmp remove next ->, test_sample_num = 150):
    dataset_paths = list((MAIN_DATASET_PATH / dataset_type).glob(f'*{dataset_name}*'))
    for dataset_path in dataset_paths:
        dataset = np.load(dataset_path)
        np.random.seed(SEED)
        # get indexes of normal sampels (0's) and anomalies (1's)
        normal_idx = np.where(dataset['y'] == 0)[0]
        anomalies_idx = np.where(dataset['y'] == 1)[0]
        # get anumber of random indexes of normal samples the same as the number of anomalies
        random_idx_from_normal_set = np.random.choice(range(normal_idx.shape[0]), len(anomalies_idx), replace=False)
        # Creating a boolean mask for all elements (default True)
        mask = np.ones(len(normal_idx), dtype=bool)
        # Set the mask to False for the excluded indexes
        mask[random_idx_from_normal_set] = False

        train_idx = normal_idx[mask]
        test_idx = np.concatenate((normal_idx[~mask], anomalies_idx))
        data_train = dataset['X'][train_idx]
        gt_train_s = dataset['y'][train_idx]
        data_test = dataset['X'][test_idx]
        gt_test_s = dataset['y'][test_idx]

        # random_idx_s = np.random.choice(range(dataset['X'].shape[0]), dataset['X'].shape[0], replace=False)
        # data_train = dataset['X'][random_idx_s[:-test_sample_num]]
        # gt_train_s = dataset['y'][random_idx_s[:-test_sample_num]]
        # data_test = dataset['X'][random_idx_s[-test_sample_num:]]
        # gt_test_s = dataset['y'][random_idx_s[-test_sample_num:]]
        # # Remove anomalies from the train set
        # data_train = data_train[gt_train_s == 0]

        if knnn_configs is None:
            knnn_configs = {
                "number_of_neighbors": 3, # 5 
                "number_of_neighbors_of_neighbors": 25, #  100, # 15, 
                "set_size": 4,  # 2
            }
        do_multi_run_random_reorder = False
        if not do_multi_run_random_reorder:
            knnn = KNNN(**knnn_configs)
            knnn.fit(data_train) 
            knnn_res, knn_dict = knnn(data_test, return_nearest_neigbours_results=True)
            knn_res = knn_dict['knn_distance'].mean(1)
            num_of_runs = 1
        else:
            raise NotImplementedError("do_multi_run_random_reorder")
            knnn_configs['reorder_method'] = 'random'
            num_of_runs = 10
            knn_res, knnn_res = None, None
            for _ in tqdm(range(num_of_runs), desc=f"Running {num_of_runs} random reorder runs"):
                knnn = KNNN(**knnn_configs)
                knnn.fit(data_train)
                knnn_res_curr, knn_res_curr = knnn(data_test)
                knn_res = np.concatenate((knn_res, np.expand_dims(knn_res_curr, 1)), axis=1) if knn_res is not None else np.expand_dims(knn_res_curr, 1)
                knnn_res = np.concatenate((knnn_res, np.expand_dims(knnn_res_curr, 1)), axis=1) if knnn_res is not None else np.expand_dims(knnn_res_curr, 1)
            knn_res = np.mean(knn_res, axis=1)
            knnn_res = np.mean(knnn_res, axis=1)

        # Compere scores
        roc_auc_score_knn = roc_auc_score(gt_test_s, knn_res)
        roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_res)
        average_precision_score_knn = average_precision_score(gt_test_s, knn_res)
        average_precision_score_knnn = average_precision_score(gt_test_s, knnn_res)

        # Document results
        dataset_name = f'{dataset_type}_{dataset_path.stem}'
        results_dict = {
            "dataset": dataset_name,
            "test_sample_num": len(data_test),
            "train_sample_num": len(data_train),
            "number_of_neighbors": knnn.number_of_neighbors,
            "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
            "set_size": knnn.set_size,
            "seed": SEED,
            "knnn": roc_auc_score_knnn,
            "knn": roc_auc_score_knn,
            "knnn_ap": average_precision_score_knnn,
            "knn_ap": average_precision_score_knn,
            "do_multi_run_random_reorder": do_multi_run_random_reorder,
            "num_of_runs": num_of_runs,
        }
        document_results(results_dict)

        # assert roc_auc_score_knn > 0.5, f"something is wrong with the knn on {dataset_name} dataset (knnn={roc_auc_score_knn})"
        # assert roc_auc_score_knnn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={roc_auc_score_knnn})"
        assert roc_auc_score_knnn >= roc_auc_score_knn, f"KNNN is not better KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
        # assert average_precision_score_knn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={average_precision_score_knn})"
        # assert average_precision_score_knnn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={average_precision_score_knnn})"
        assert average_precision_score_knnn >= average_precision_score_knn, f"KNNN is not better than KNN by AP metric on {dataset_name} dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"

        logging.info(f"KNNN is better than KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")
        print(f"KNNN is better than KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")

# def run_test_on_ADB_dataset_parts(dataset_type, dataset_name, knnn_configs=None):
#     size_of_part = 3
#     dataset_paths = list((MAIN_DATASET_PATH / dataset_type).glob(f'*{dataset_name}*'))
#     for dataset_path in dataset_paths:
#         dataset = np.load(dataset_path)
#         test_sample_num = 150
#         np.random.seed(SEED)
#         random_idx_s = np.random.choice(range(dataset['X'].shape[0]), dataset['X'].shape[0], replace=False)
#         data_train = dataset['X'][random_idx_s[:-test_sample_num]]
#         gt_train_s = dataset['y'][random_idx_s[:-test_sample_num]]
#         data_test = dataset['X'][random_idx_s[-test_sample_num:]]
#         gt_test_s = dataset['y'][random_idx_s[-test_sample_num:]]
#         # Remove anomalies from the train set
#         data_train = data_train[gt_train_s == 0]
#         num_of_parts = data_train.shape[1] // size_of_part
#         if knnn_configs is None:
#             knnn_configs = {
#                 "number_of_neighbors": 1, # 100, # 3
#                 "number_of_neighbors_of_neighbors": 15,
#                 "set_size": size_of_part,  # 2
#             }

#         knn_res, knnn_res = None, None
#         for _ in tqdm(range(num_of_parts), desc=f"Running knnn in parts", leave=False):
#             knnn = KNNN(**knnn_configs)
#             random_feat_idx_s = np.random.choice(range(data_train.shape[1]), size_of_part, replace=False)
#             data_train_part = data_train[:, random_feat_idx_s]
#             data_test_part = data_test[:, random_feat_idx_s]
#             knnn.fit(data_train_part)
#             res_knnn_curr, res_knn_curr = knnn(data_test_part, return_nearest_neigbours_results=True)
#             knn_res = np.concatenate((knn_res, np.expand_dims(res_knn_curr, 1)), axis=1) if knn_res is not None else np.expand_dims(res_knn_curr, 1)
#             knnn_res = np.concatenate((knnn_res, np.expand_dims(res_knnn_curr, 1)), axis=1) if knnn_res is not None else np.expand_dims(res_knnn_curr, 1)
#         knn_res = np.mean(knn_res, axis=1)
#         knnn_res = np.mean(knnn_res, axis=1)


#         # Compere scores
#         roc_auc_score_knn = roc_auc_score(gt_test_s, knn_res)
#         roc_auc_score_knnn = roc_auc_score(gt_test_s, knnn_res)
#         average_precision_score_knn = average_precision_score(gt_test_s, knn_res)
#         average_precision_score_knnn = average_precision_score(gt_test_s, knnn_res)

#         # Document results
#         dataset_name = f'{dataset_type}_{dataset_path.stem}'
#         results_dict = {
#             "dataset": dataset_name,
#             "test_sample_num": test_sample_num,
#             "train_sample_num": len(data_train),
#             "number_of_neighbors": knnn.number_of_neighbors,
#             "number_of_neighbors_of_neighbors": knnn.number_of_neighbors_of_neighbors,
#             "set_size": knnn.set_size,
#             "seed": SEED,
#             "knnn": roc_auc_score_knnn,
#             "knn": roc_auc_score_knn,
#             "knnn_ap": average_precision_score_knnn,
#             "knn_ap": average_precision_score_knn,
#             "do_parts": True,
#             "num_of_parts": num_of_parts,
#         }
#         document_results(results_dict)

#         # assert roc_auc_score_knn > 0.5, f"something is wrong with the knn on {dataset_name} dataset (knnn={roc_auc_score_knn})"
#         # assert roc_auc_score_knnn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={roc_auc_score_knnn})"
#         assert roc_auc_score_knnn >= roc_auc_score_knn, f"parts KNNN is not better KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})"
#         # assert average_precision_score_knn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={average_precision_score_knn})"
#         # assert average_precision_score_knnn > 0.5, f"something is wrong with the knnn on {dataset_name} dataset (knnn={average_precision_score_knnn})"
#         assert average_precision_score_knnn >= average_precision_score_knn, f"KNNN is not better than KNN by AP metric on {dataset_name} dataset (knnn={average_precision_score_knnn}, knn={average_precision_score_knn})"

#         logging.info(f" parts KNNN is better than KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")
#         print(f"KNNN is better than KNN by AUROC metric on {dataset_name} dataset (knnn={roc_auc_score_knnn}, knn={roc_auc_score_knn})")


# Classical
# 1
def test_knn_vs_knnn_ADB_Classical_ALOI():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='ALOI')

# def test_knn_vs_knnn_ADB_Classical_ALOI_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='Classical', dataset_name='ALOI')

# 2
def test_knn_vs_knnn_ADB_Classical_annthyroid():
    knnn_configs = {
    "number_of_neighbors": 200, 
    "number_of_neighbors_of_neighbors": 100,
    "set_size": 3,  
    }
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='annthyroid', knnn_configs=knnn_configs)
# 3    
def test_knn_vs_knnn_ADB_Classical_backdoor():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='backdoor')
# 4    
def test_knn_vs_knnn_ADB_Classical_breastw():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='breastw')
# 5    
def test_knn_vs_knnn_ADB_Classical_campaign():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='campaign')
# 6    
def test_knn_vs_knnn_ADB_Classical_cardio():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='cardio')
# 7    
def test_knn_vs_knnn_ADB_Classical_Cardiotocography():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Cardiotocography')
# 8   
def test_knn_vs_knnn_ADB_Classical_celeba():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='celeba')
# 9    
def test_knn_vs_knnn_ADB_Classical_census():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='census')
# 10    
def test_knn_vs_knnn_ADB_Classical_CoverType():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='CoverType')
# 11    
def test_knn_vs_knnn_ADB_Classical_DonorsChoose():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='DonorsChoose')
# 12
def test_knn_vs_knnn_ADB_Classical_fault():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='fault')
# 13
def test_knn_vs_knnn_ADB_Classical_fraud():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='fraud')
# 14
def test_knn_vs_knnn_ADB_Classical_glass():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='glass')
# 15
def test_knn_vs_knnn_ADB_Classical_Hepatitis():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Hepatitis')
# 16
def test_knn_vs_knnn_ADB_Classical_http():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='http')
# 17
def test_knn_vs_knnn_ADB_Classical_InternetAds():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='InternetAds')
# 18
def test_knn_vs_knnn_ADB_Classical_Ionosphere():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Ionosphere')
# 19
def test_knn_vs_knnn_ADB_Classical_Landsat():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Landsat')
# 20
def test_knn_vs_knnn_ADB_Classical_letter():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='letter')
# 21
def test_knn_vs_knnn_ADB_Classical_Lymphography():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Lymphography')
# 22
def test_knn_vs_knnn_ADB_Classical_magic():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='magic.gamma')
# 23
def test_knn_vs_knnn_ADB_Classical_Mammography():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='mammography') 
# 24
def test_knn_vs_knnn_ADB_Classical_mnist():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='mnist')
# 25
def test_knn_vs_knnn_ADB_Classical_musk():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='musk')
# 26
def test_knn_vs_knnn_ADB_Classical_optdigits():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='optdigits')
# 27
def test_knn_vs_knnn_ADB_Classical_PageBlocks():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='PageBlocks')
# 28
def test_knn_vs_knnn_ADB_Classical_pedigits():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='pendigits')
# 29
def test_knn_vs_knnn_ADB_Classical_Pima():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Pima')
# 30
def test_knn_vs_knnn_ADB_Classical_satellite():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='satellite')
# 31
def test_knn_vs_knnn_ADB_Classical_satimage():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='satimage-2')
# 32
def test_knn_vs_knnn_ADB_Classical_shuttle():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='shuttle')
# 33
def test_knn_vs_knnn_ADB_Classical_skin():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='skin')
# 34
def test_knn_vs_knnn_ADB_Classical_samtp():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='smtp')
# 35
def test_knn_vs_knnn_ADB_Classical_SpamBase():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='SpamBase')
# 36
def test_knn_vs_knnn_ADB_Classical_speech():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='speech')
# 37
def test_knn_vs_knnn_ADB_Classical_Stamps():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Stamps')
# 38
def test_knn_vs_knnn_ADB_Classical_thyroid():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='thyroid')
# 39
def test_knn_vs_knnn_ADB_Classical_vertebral():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='vertebral')
# 40
def test_knn_vs_knnn_ADB_Classical_Vowels():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Vowels')
# 41
def test_knn_vs_knnn_ADB_Classical_Waveform():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Waveform')
# 42
def test_knn_vs_knnn_ADB_Classical_WBC():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='WBC')
# 43
def test_knn_vs_knnn_ADB_Classical_WDBC():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='WDBC')
# 44
def test_knn_vs_knnn_ADB_Classical_Wilt():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='Wilt')
# 45
def test_knn_vs_knnn_ADB_Classical_wine():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='wine')
# 46
def test_knn_vs_knnn_ADB_Classical_WPBC():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='WPBC')
# 47
def test_knn_vs_knnn_ADB_Classical_yeast():
    run_test_on_ADB_dataset(dataset_type='Classical', dataset_name='yeast')
    

# RESNET18
def test_knn_vs_knnn_ADB_ResNet18_bottle():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_bottle')

def test_knn_vs_knnn_ADB_ResNet18_MVTec_cable():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_cable')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_capsule():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_capsule')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_carpet():
    # TODO tmp remove
    # knnn_configs = {
    #     "number_of_neighbors": 3, 
    #     "number_of_neighbors_of_neighbors": 25,
    #     "set_size": 4,  
    # }
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_carpet')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_grid():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_grid') 
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_hazelnut():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_hazelnut')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_leather():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_leather')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_metal_nut():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_metal_nut')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_pill():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_pill')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_screw():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_screw')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_tile():
    # knnn_configs = { # TODO tmp remove
    #     "number_of_neighbors": 3, 
    #     "number_of_neighbors_of_neighbors": 25,
    #     "set_size": 4,  
    # }
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_tile')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_toothbrush():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_toothbrush')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_transistor():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_transistor')
    
def test_knn_vs_knnn_ADB_ResNet18_MVTec_wood():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_wood')

def test_knn_vs_knnn_ADB_ResNet18_MVTec_zipper():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MVTec-AD_zipper')

# cifar10
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_0')

def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_1')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_2')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_3')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_4')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_5')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_6')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_7')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_8')
    
def test_knn_vs_knnn_ADB_ResNet18_CIFAR10_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='CIFAR10_9')
    
# fashionMNIST
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_0')

def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_1')
    
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_2')
    
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_3')

def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_4')

def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_5')

def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_6')
    
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_7')
    
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_8')
    
def test_knn_vs_knnn_ADB_ResNet18_FashionMNIST_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='FashionMNIST_9')

# MNIST-C
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_brightness():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_brightness')

def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_canny_edges():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_canny_edges')

def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_dotted_line():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_dotted_line')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_fog():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_fog')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_glass_blur():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_glass_blur')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_identity():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_identity')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_impulse_noise():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_impulse_noise')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_motion_blur():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_motion_blur')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_rotate():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_rotate')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_scale():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_scale')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_shear():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_shear')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_shot_noise():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_shot_noise')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_spatter():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_spatter')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_stripe():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_stripe')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_translate():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_translate')
    
def test_knn_vs_knnn_ADB_ResNet18_MNIST_C_zigzag():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='MNIST-C_zigzag')
    
# SVHN
def test_knn_vs_knnn_ADB_ResNet18_SVHN_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_0')

def test_knn_vs_knnn_ADB_ResNet18_SVHN_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_1')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_2')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_3')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_4')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_5')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_6')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_7')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_8')
    
def test_knn_vs_knnn_ADB_ResNet18_SVHN_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ResNet18', dataset_name='SVHN_9')

# ViT
# MVTec
def test_knn_vs_knnn_ADB_ViT_MVTec_bottle():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_bottle')

def test_knn_vs_knnn_ADB_ViT_MVTec_cable():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_cable')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_capsule():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_capsule')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_carpet():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_carpet')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_grid():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_grid')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_hazelnut():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_hazelnut')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_leather():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_leather')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_metal_nut():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_metal_nut')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_pill():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_pill')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_screw():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_screw')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_tile():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_tile')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_toothbrush():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_toothbrush')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_transistor():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_transistor')
    
def test_knn_vs_knnn_ADB_ViT_MVTec_wood():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_wood')

def test_knn_vs_knnn_ADB_ViT_MVTec_zipper():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MVTec-AD_zipper')

# cifar10
def test_knn_vs_knnn_ADB_ViT_CIFAR10_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_0')

def test_knn_vs_knnn_ADB_ViT_CIFAR10_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_1')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_2')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_3')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_4')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_5')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_6')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_7')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_8')
    
def test_knn_vs_knnn_ADB_ViT_CIFAR10_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='CIFAR10_9')
    
# fashionMNIST
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_0')

def test_knn_vs_knnn_ADB_ViT_FashionMNIST_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_1')
    
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_2')
    
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_3')

def test_knn_vs_knnn_ADB_ViT_FashionMNIST_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_4')

def test_knn_vs_knnn_ADB_ViT_FashionMNIST_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_5')

def test_knn_vs_knnn_ADB_ViT_FashionMNIST_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_6')
    
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_7')
    
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_8')
    
def test_knn_vs_knnn_ADB_ViT_FashionMNIST_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='FashionMNIST_9')

# MNIST-C
def test_knn_vs_knnn_ADB_ViT_MNIST_C_brightness():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_brightness')

def test_knn_vs_knnn_ADB_ViT_MNIST_C_canny_edges():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_canny_edges')

def test_knn_vs_knnn_ADB_ViT_MNIST_C_dotted_line():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_dotted_line')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_fog():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_fog')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_glass_blur():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_glass_blur')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_identity():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_identity')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_impulse_noise():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_impulse_noise')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_motion_blur():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_motion_blur')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_rotate():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_rotate')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_scale():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_scale')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_shear():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_shear')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_shot_noise():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_shot_noise')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_spatter():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_spatter')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_stripe():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_stripe')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_translate():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_translate')
    
def test_knn_vs_knnn_ADB_ViT_MNIST_C_zigzag():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='MNIST-C_zigzag')
    
# SVHN
def test_knn_vs_knnn_ADB_ViT_SVHN_0():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_0')

def test_knn_vs_knnn_ADB_ViT_SVHN_1():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_1')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_2():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_2')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_3():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_3')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_4():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_4')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_5():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_5')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_6():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_6')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_7():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_7')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_8():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_8')
    
def test_knn_vs_knnn_ADB_ViT_SVHN_9():
    run_test_on_ADB_dataset(dataset_type='CV_by_ViT', dataset_name='SVHN_9')


# BERT
# 20news
def test_knn_vs_knnn_ADB_BERT_20news_0():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_0')

def test_knn_vs_knnn_ADB_BERT_20news_1():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_1')
    
def test_knn_vs_knnn_ADB_BERT_20news_2():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_2')
    
def test_knn_vs_knnn_ADB_BERT_20news_3():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_3')
    
def test_knn_vs_knnn_ADB_BERT_20news_4():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_4')
    
def test_knn_vs_knnn_ADB_BERT_20news_5():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='20news_5')

# agnews
def test_knn_vs_knnn_ADB_BERT_agnews_0():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='agnews_0')

def test_knn_vs_knnn_ADB_BERT_agnews_1():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='agnews_1')
    
def test_knn_vs_knnn_ADB_BERT_agnews_2():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='agnews_2')
    
def test_knn_vs_knnn_ADB_BERT_agnews_3():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='agnews_3')
     
def test_knn_vs_knnn_ADB_BERT_amazon():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='amazon')
    
def test_knn_vs_knnn_ADB_BERT_yelp():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='yelp')

def test_knn_vs_knnn_ADB_BERT_imdb():
    run_test_on_ADB_dataset(dataset_type='NLP_by_BERT', dataset_name='imdb')

# # agnews parts
# def test_knn_vs_knnn_ADB_BERT_agnews_0_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='agnews_0')

# def test_knn_vs_knnn_ADB_BERT_agnews_1_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='agnews_1')
    
# def test_knn_vs_knnn_ADB_BERT_agnews_2_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='agnews_2')
    
# def test_knn_vs_knnn_ADB_BERT_agnews_3_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='agnews_3')
     
# def test_knn_vs_knnn_ADB_BERT_amazon_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='amazon')
    
# def test_knn_vs_knnn_ADB_BERT_yelp_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='yelp')

# def test_knn_vs_knnn_ADB_BERT_imdb_parts():
#     run_test_on_ADB_dataset_parts(dataset_type='NLP_by_BERT', dataset_name='imdb')




# RoBERTa
# 20news
def test_knn_vs_knnn_ADB_RoBERTa_20news_0():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_0')

def test_knn_vs_knnn_ADB_RoBERTa_20news_1():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_1')
    
def test_knn_vs_knnn_ADB_RoBERTa_20news_2():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_2')
    
def test_knn_vs_knnn_ADB_RoBERTa_20news_3():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_3')
    
def test_knn_vs_knnn_ADB_RoBERTa_20news_4():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_4')
    
def test_knn_vs_knnn_ADB_RoBERTa_20news_5():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='20news_5')
    
# agnews  
def test_knn_vs_knnn_ADB_RoBERTa_agnews_0():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='agnews_0')

def test_knn_vs_knnn_ADB_RoBERTa_agnews_1():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='agnews_1')
    
def test_knn_vs_knnn_ADB_RoBERTa_agnews_2():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='agnews_2')
    
def test_knn_vs_knnn_ADB_RoBERTa_agnews_3():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='agnews_3')

def test_knn_vs_knnn_ADB_RoBERTa_amazon():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='amazon')

def test_knn_vs_knnn_ADB_RoBERTa_yelp():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='yelp')

def test_knn_vs_knnn_ADB_RoBERTa_imdb():
    run_test_on_ADB_dataset(dataset_type='NLP_by_RoBERTa', dataset_name='imdb')

 


    
if __name__ == '__main__':
    test_knn_vs_knnn_ADB_ResNet18_bottle()
    # test_knn_vs_knnn_ADB_RoBERTa_20news()
    # test_knn_vs_knnn_ADB_RoBERTa_agnews_0()
    # test_knn_vs_knnn_ADB_BERT_agnews_0_parts()