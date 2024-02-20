from pathlib import Path
import logging
from typing import List, Tuple, Union, Optional

from PIL import Image
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_swiss_roll, make_blobs

synthetic_dataset_types = ['moons', 'circles', 'swiss_roll', 'blobs']
def create_samples(normal_num_of_samples: int, 
                   noises: Optional[float]=None, 
                   seed: Optional[int] = 0, 
                   data_types: Optional[List[str]]=['moons', 'blobs']):
    """_summary_
    Create synthetic data, with uniform grid test data.

    Args:
        nornal_num_of_samples (int): create this number of normal train samples
        noises (Optional[List[float]], optional): add noise to each of the data_types with noises stds. Defaults to None.
        seed (Optional[int], optional): set the seed. Defaults to 0.
        data_types (Optional[List[str]], optional): create these dataset types. Defaults to ['moons', 'blobs'].

    Raises:
        NotImplementedError: data_types not implemented

    Returns:
        xn_train_s (np.array), xn_test_s (np.array), xa_test_s (np.array): xn_train_s normal train samples, xn_test_s normal test samples, xa_test_s anomalies test samples 
    """
    logging.info(f'creating synthetic data with {normal_num_of_samples} normal sampels, noises={noises}, seed={seed}, data_types={data_types}')
    np.random.seed(seed)
    xn_s = []
    xn_test_s = []
    cls_s = []
    cls_test_s = []
    test_num_of_samples = 100_000 # befor sampling
    if noises is None:
        noises = [None] * len(data_types)
    for dataset_type, noise in zip(data_types, noises):
        noise = noise if noise is not None else 0.0
        tot_samples = normal_num_of_samples + test_num_of_samples
        if dataset_type == 'moons':
            all_xn_s, all_cls_s, = make_moons(n_samples=tot_samples, shuffle=True, noise=noise, random_state=seed) 
            all_xn_s *= 10
        elif dataset_type == 'blobs':
            all_xn_s, all_cls_s = make_blobs(n_samples=tot_samples, shuffle=True, centers=3, cluster_std=[0.5, 2., 4])
            all_xn_s, all_cls_s = make_blobs(n_samples=tot_samples, shuffle=True, centers=[[10,0],[0,10],[-10,-10]], cluster_std=[0.5, 2., 4])
            all_xn_s *= 10
        elif dataset_type == 'circles':
            all_xn_s, all_cls_s = make_circles(n_samples=tot_samples, shuffle=True, noise=noise, random_state=seed)
            all_xn_s *= 10
        elif dataset_type == 'swiss_roll':
            all_xn_s, all_cls_s = make_swiss_roll(n_samples=tot_samples, noise=noise, random_state=seed)
            all_xn_s = all_xn_s[:, [0,2]]
            all_xn_s *= 10
        elif dataset_type.startswith('draw'):
            img_name = '_'.join(dataset_type.split('_')[1:]) + '.png'
            imgs_dir_path = Path.cwd() / 'drawings_points' / 'drawings' 
            img_path = imgs_dir_path / img_name
            all_xn_s = draw_to_points(img_path=img_path, num_of_points_to_sample=tot_samples)
            all_xn_s = all_xn_s[:, [0,1]]
            all_cls_s = np.zeros(all_xn_s.shape[0])
            all_xn_s *= 10
        else:
            logging.error(f'unknown dataset_type={dataset_type}')
            raise NotImplementedError('dataset_type is not supported')
        xn_s += [all_xn_s[:normal_num_of_samples, :]]
        xn_test_s += [all_xn_s[normal_num_of_samples:, :]]
        cls_s += [all_cls_s[:normal_num_of_samples]]
        cls_test_s += [all_cls_s[normal_num_of_samples:]]

    xn_s = np.concatenate(xn_s, axis=0)
    xn_test_s = np.concatenate(xn_test_s, axis=0)
    cls_s = np.concatenate(cls_s, axis=0)
    cls_test_s = np.concatenate(cls_test_s, axis=0)
    
    random_choice_ind = np.random.choice(range(len(xn_s)), normal_num_of_samples, replace=False)
    xn_train_s, cls_train_s = xn_s[random_choice_ind, :], cls_s[random_choice_ind]
    
    x_min, x_max = xn_train_s[:, 0].min(), xn_train_s[:, 0].max() 
    y_min, y_max = xn_train_s[:, 1].min(), xn_train_s[:, 1].max() 
    len_b = max(x_max - x_min, y_max - y_min)
    mid_x, mid_y = (x_max + x_min) / 2, (y_max + y_min) / 2
    x_max, x_min = mid_x + len_b / 2, mid_x - len_b / 2
    y_max, y_min = mid_y + len_b / 2, mid_y - len_b / 2
    margin = len_b / 4
    x_min, x_max = xn_train_s[:, 0].min() - margin, xn_train_s[:, 0].max() + margin
    y_min, y_max = xn_train_s[:, 1].min() - margin, xn_train_s[:, 1].max() + margin

    step_size_x = (x_max - x_min) / 200
    step_size_y = (y_max - y_min) / 200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size_x), np.arange(y_min, y_max, step_size_y))
    xa_test_s = np.concatenate((np.expand_dims(xx.ravel(), axis=1), np.expand_dims(yy.ravel(), axis=1)), axis=1)
        
    # make test normal the same size as anomaly data
    random_choice_ind_test = np.random.choice(range(xn_test_s.shape[0]), xa_test_s.shape[0], replace=False)
    xn_test_s, cls_test_s = xn_test_s[random_choice_ind_test, :], cls_test_s[random_choice_ind_test]
    return xn_train_s, xn_test_s, xa_test_s


def plot_and_save_synthetic(train_n, test_n, test_a, image_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.clf()
    sns.scatterplot(x=test_a[:, 0], y=test_a[:, 1], color='red', label=f'anomaly_test (# {test_a.shape[0]})')
    sns.scatterplot(x=test_n[:, 0], y=test_n[:, 1], color='green', label=f'normal_test (# {test_n.shape[0]})')
    sns.scatterplot(x=train_n[:, 0], y=train_n[:, 1], color='blue', label=f'normal_train (# {train_n.shape[0]})')
    save_path = Path('tests') / 'images'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path/(image_name + ".png"))
    

def draw_to_points(img_path: str, num_of_points_to_sample: int):
    """_summary_
    convet a drawing to points sampled in a distribution that is proportional to the colore of an image

    Args:
        img_path(str): path to an image
        num_of_points_to_sample (int): num of samples to return

    Returns:
        _type_: _description_
    """
    # load the image 
    img = Image.open(img_path)
    # convert to grayscale
    img = img.convert('L')
    # convert to numpy array
    img_array = np.asarray(img)
    # normalize the image
    img_array = img_array - img_array.min()
    img_array = img_array / img_array.max()
    # # reverse black and white
    img_array = 1 - img_array
    pixel_probability = img_array.flatten() / img_array.sum()

    # sampler
    points_chosen = np.random.choice(img_array.size, size=num_of_points_to_sample, p=pixel_probability)
    points_img = [indx in points_chosen for indx in range(img_array.shape[0] * img_array.shape[1])]
    points_img = np.array(points_img).reshape(img_array.shape)

    y, x = np.where(points_img)
    res_points = np.stack((x, -y), axis=1)
    res_points = res_points - res_points.min(axis=0)
    res_points = res_points / (res_points.max(axis=0)).max()
    # center the points
    res_points = res_points - res_points.mean(axis=0)

    np.random.shuffle(res_points)
    return res_points