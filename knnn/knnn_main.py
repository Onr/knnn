import time
import logging

from tqdm.auto import tqdm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster

import faiss


def compute_correlation_matrix(data):
    """
    Computes the correlation matrix for a given 2D NumPy array.
    
    :param data: A 2D NumPy array of size [n, m] where n is the number of samples
                 and m is the number of features.
    :return: The correlation matrix of size [m, m].
    """
    # Standardize each feature (column) to have mean 0 and variance 1
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

    # Compute the correlation matrix
    correlation_matrix = np.dot(standardized_data.T, standardized_data) / (data.shape[0] - 1)

    return correlation_matrix

    
def hierarchical_clustering(correlation_matrix, do_plot=False):
    # Transform the correlation matrix to a distance matrix
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Ensure the diagonal elements are zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Perform hierarchical/agglomerative clustering
    Z = linkage(squareform(distance_matrix), method='average')
    return Z


def equal_size_clustering(Z, m, target_size):
    """
    Divides features into approximately equal-sized clusters.
    
    :param Z: Linkage matrix from hierarchical clustering.
    :param m: Total number of features.
    :param target_size: Desired number of features in each cluster.
    :return: Array indicating cluster membership for each feature.
    """
    
    if m % target_size != 0:
        raise ValueError("Number of features target_size must be a divisor of number of features m")

    k = m // target_size

    # Initial clustering
    initial_clusters = fcluster(Z, k, criterion='maxclust')
    # If the initial clustering does not result in k clusters, we need to adjust it
    for cluster_ind in range(1, k + 1):
        cluster_size = np.sum(initial_clusters == cluster_ind) 
        if cluster_size == 0:
            # find non empty cluster
            for i in range(1, k + 1):
                if i != cluster_ind:
                    cluster_size_other = np.sum(initial_clusters == i) 
                    if cluster_size_other > target_size:
                        # change an elemnet to be of the other 
                        initial_clusters[np.where(initial_clusters == i)[0][0]] = cluster_ind
                        break
    
    # Target size for each cluster
    target_size = m // k

    # Create a list to hold the merge level for each feature
    merge_levels = np.zeros(2 * m - 1)

    # Fill in the merge levels from the linkage matrix
    for i in range(m - 1):
        cluster_formed = int(Z[i, 0]), int(Z[i, 1])
        for j in cluster_formed:
            merge_levels[j] = Z[i, 2]

    # Adjustment for equal size
    for cluster_id in range(1, k + 1):
        while np.sum(initial_clusters == cluster_id) > target_size:
            # Find the loosest feature in this cluster
            indices_in_cluster = np.where(initial_clusters == cluster_id)[0]
            loosest_feature = indices_in_cluster[np.argmax(merge_levels[indices_in_cluster])]

            # Find the closest cluster to move the loosest feature into
            closest_cluster = None
            min_distance = np.inf
            for i in range(1, k + 1):
                if i != cluster_id and np.sum(initial_clusters == i) < target_size:
                    indices_in_other_cluster = np.where(initial_clusters == i)[0]
                    if indices_in_other_cluster.size > 0:
                        cluster_distances = merge_levels[indices_in_other_cluster + m - 1]  # Adjust indices for clusters
                        distance = np.abs(merge_levels[loosest_feature + m - 1] - cluster_distances.mean())  # Adjust index for current feature
                        if distance < min_distance:
                            min_distance = distance
                            closest_cluster = i

            # Move the feature to the closest cluster
            if closest_cluster is not None:
                initial_clusters[loosest_feature] = closest_cluster

    return initial_clusters


def get_features_clusters(data, target_size):
    """
    Performs hierarchical clustering on features and returns the clusters.
    
    :param data: Data matrix with features in columns.
    :param target_size: Desired number of features in each cluster.
    :return: Array indicating cluster membership for each feature.
    """
    if data.shape[1] < target_size:
        raise ValueError("Number of target_size must be less than or equale the number of features in the data")
    if data.shape[1] == target_size or target_size <= 1:
        return np.zeros(data.shape[1])
    correlation_matrix = compute_correlation_matrix(data)
    Z = hierarchical_clustering(np.abs(correlation_matrix))
    Z = np.where(Z < 0, 0, Z)
    m = data.shape[1]
    clusters = equal_size_clustering(Z, m, target_size)
    return clusters


def get_features_sets(data, clusters):
    sets = []
    for cluster in np.unique(clusters):
        sets.append(np.ascontiguousarray(data[:, clusters == cluster]))
    return sets


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def get_features_knn_faiss(data_1, data_2=None, dist_func='cosine', k=25, do_gpu=False, index=None, return_index=False, return_centers=False):
    # get knn from data_1 vecs to data_2 vecs the return index are of data_2 vecs
    data_1 = data_1.astype('float32')
    if data_2 is not None:
        data_2 = data_2.astype('float32')
    if index is None:
        # Initialize the index
        if do_gpu:
            res = faiss.StandardGpuResources()
        if dist_func == 'cosine':
            # Normalize each sample to unit length
            data_1 = normalize_vectors(data_1)
            data_2 = normalize_vectors(data_2)
            # Faiss Inner Product (IP) index is equivalent to cosine similarity if data is normalized
            index = faiss.IndexFlatIP(data_2.shape[1]) if not do_gpu else faiss.GpuIndexFlatIP(res, data_1.shape[1])
        elif dist_func == 'euclidean':
            index = faiss.IndexFlatL2(data_2.shape[1]) if not do_gpu else faiss.GpuIndexFlatL2(res, data_1.shape[1])
        else:
            raise ValueError("Invalid distance function")
    if data_2 is not None:
        index.reset()
        index.add(data_2)

    # Search for nearest neighbors
    D, I = index.search(data_1, k)

    # Convert cosine similarity to distance
    if dist_func == 'cosine':
        D = 1 - D
    
    return_values = (D, I) 

    if return_centers and data_2 is not None:
        centroids = np.zeros((data_2.shape), dtype='float32')
        for i, indices in enumerate(I):
            # Exclude the first index to remove the point itself from its neighbors
            neighbors = data_2[indices[1:]]  # Skip the first neighbor
            centroid = np.mean(neighbors, axis=0)
            centroids[i] = centroid
        return_values += (centroids,)

    if return_index:
        return_values += (index,) 

    return return_values


def compute_invers_covariances_and_centers(embedding_matrix, indices_matrix, return_eigen_vectors=False, use_means_as_return__centers=False):
    n, m = embedding_matrix.shape
    n, k = indices_matrix.shape
    
    centers = np.zeros((n, m))
    inverse_covariance_matries = np.zeros((n, m, m))
    if return_eigen_vectors:
        eig_vec_size = min(m, k)
        eigenvalues = np.zeros((n, eig_vec_size))
        eigenvectors = np.zeros((n, eig_vec_size, m))
    
    for i in tqdm(range(n), desc='Computing eigenvectors and eigenvalues',leave=False):
        # Select the embedding of the sample and its nearest neighbors
        neighbors = indices_matrix[i]
        sub_matrix = embedding_matrix[neighbors, :]

        # compute the center of the neighbors
        centers[i] = sub_matrix[0, :] if not use_means_as_return__centers else np.mean(sub_matrix, axis=0) 
        
        # Compute the covariance matrix for the sample and its neighbors
        cov_matrix = np.cov(sub_matrix, rowvar=False)

        # Step 3: Calculate the inverse of the covariance matrix
        try:
            inverse_covariance_matrix = np.linalg.inv(cov_matrix)
        except:
            # If the covariance matrix is singular, use just an identity matrix
                inverse_covariance_matrix = np.eye(cov_matrix.shape[0])

        if return_eigen_vectors:
            # Compute eigenvalues and eigenvectors
            try:
                vals, vecs = np.linalg.eig(cov_matrix)
            except:
                # If the covariance matrix is singular, use just an identity matrix
                vals = np.ones(cov_matrix.shape[0])
                vecs = np.eye(cov_matrix.shape[0])
        
        # Store the results
        inverse_covariance_matries[i, :, :] = inverse_covariance_matrix
        if return_eigen_vectors:
            eigenvalues[i, :len(vals)] = vals
            eigenvectors[i, :len(vecs), :] = vecs.T

    if return_eigen_vectors:
        return centers, inverse_covariance_matries, eigenvalues, eigenvectors
    else:
        return centers, inverse_covariance_matries

        
class KNNN:
    def __init__(self, number_of_neighbors=3, number_of_neighbors_of_neighbors=25, set_size=5, distance_function='cosine') -> None:
        self.number_of_neighbors = number_of_neighbors
        self.number_of_neighbors_of_neighbors = number_of_neighbors_of_neighbors
        self.set_size = set_size
        self.distance_function = distance_function
        self.fitted = False
        self.droped_features_with_zero_std = None 
        self.values_of_droped_features_with_zero_std = None
        
    def fit(self, data):
        logging.info('Fitting knnn model')
        start_time = time.time()
        self.droped_features_with_zero_std = np.where(np.std(data, axis=0) == 0)[0]
        self.values_of_droped_features_with_zero_std = data[:, self.droped_features_with_zero_std][0]
        self.data = data if self.droped_features_with_zero_std is None else np.delete(data, self.droped_features_with_zero_std, axis=1)
        # devide the data to sets
        # add zeros if data is not divisable by set_size
        if self.data.shape[1] % self.set_size != 0:
            self.data = self.data[:, :-(self.data.shape[1] % self.set_size)] # TODO test
            # self.num_of_zeros_features_to_add = self.set_size - self.data.shape[1] % self.set_size
            # self.data = np.concatenate([self.data, 1e-5*np.random.randn(self.data.shape[0], self.num_of_zeros_features_to_add)], axis=1)
        self.features_clusters = get_features_clusters(self.data, self.set_size)
        self.feature_sets = get_features_sets(self.data, self.features_clusters)

        self.hole_data_distances, self.hole_data_indesies, hole_data_centers, self.hole_data_index = get_features_knn_faiss(self.data, self.data, k=self.number_of_neighbors_of_neighbors, do_gpu=False, dist_func=self.distance_function, return_centers=True, return_index=True)

        self.centers_groups, self.inverse_covariance_matries_groups = [], []
        for feature_set in tqdm(self.feature_sets, desc='Computing eigenvectors and eigenvalues for each set'):
            centers, inv_mat = compute_invers_covariances_and_centers(feature_set, self.hole_data_indesies) # computer eigens for each set based on the hole data knn
            self.centers_groups.append(centers)
            self.inverse_covariance_matries_groups.append(inv_mat)
        self.centers_groups = np.stack(self.centers_groups, axis=1) # [sample, group, center_vec]
        self.inverse_covariance_matries_groups = np.stack(self.inverse_covariance_matries_groups, axis=1) # [sample, group, set_size, set_size]

        self.fitted = True
        logging.info(f'Fitting the knnn model took {time.time() - start_time:.2f} seconds')

    def __call__(self, test_data, return_nearest_neigbours_results=False) -> np.ndarray:
        assert self.fitted, 'The model is not fitted yet'
        logging.info('Calculating knnn score')
        start_time = time.time()
        test_same_values_train = None
        if self.droped_features_with_zero_std is not None:
            test_data_of_droped_features_with_zero_std = test_data[:, self.droped_features_with_zero_std]
            # check there isnt feature that wasnt in the train data
            test_same_values_train = (test_data_of_droped_features_with_zero_std != self.values_of_droped_features_with_zero_std).sum(1)
            if not np.all(test_data_of_droped_features_with_zero_std == self.values_of_droped_features_with_zero_std):
                logging.warning('There are features in the test data that wasnt in the train data')
            test_data = np.delete(test_data, self.droped_features_with_zero_std, axis=1)

        if test_data.shape[1] % self.set_size != 0:
            test_data = test_data[:, :-(test_data.shape[1] % self.set_size)] # TODO test
            # test_data = np.concatenate([test_data, 1e-5*np.random.randn(test_data.shape[0], self.num_of_zeros_features_to_add)], axis=1)

        # get knn indeies for each test data point to the hole data
        hole_test_data_distances, hole_test_data_indesies = get_features_knn_faiss(data_1=test_data, index=self.hole_data_index, k=self.number_of_neighbors, do_gpu=False, dist_func=self.distance_function, return_centers=False)

        # devide the test data into sets
        test_data_sets = get_features_sets(test_data, self.features_clusters)
        test_data_sets = np.stack(test_data_sets, axis=1) # [sample, group, data]

        # for each test data point get the mahalanobis distace to each set
        results_matrix = np.zeros((test_data.shape[0], self.number_of_neighbors, test_data_sets.shape[1]))
        for test_ind in tqdm(range(test_data.shape[0]), desc='Computing mahalanobis distance for each test sample'):
            for test_data_neigbour_ind in range(self.number_of_neighbors):
                original_data_neigbour_ind = hole_test_data_indesies[test_ind, test_data_neigbour_ind]
                for group_ind in range(test_data_sets.shape[1]):
                    # Compute the difference between the test sample and the center of the set
                    mahalanobis_distance = distance.mahalanobis(test_data_sets[test_ind, group_ind], self.centers_groups[original_data_neigbour_ind, group_ind], self.inverse_covariance_matries_groups[original_data_neigbour_ind, group_ind])
                    results_matrix[test_ind, test_data_neigbour_ind, group_ind] = mahalanobis_distance

        # combine the results of the sete and the nighebours
        test_results = results_matrix.copy()
        # opt 1 mean:
        # mean on the sets
        test_results = np.nanmean(test_results, axis=1)
        # mean on the neigbours
        test_results = np.nanmean(test_results, axis=1)

        if test_same_values_train is not None:
            test_results[test_same_values_train > 0] = np.finfo(np.float64).max
        
        logging.info(f'Calculating knnn score took {time.time() - start_time:.2f} seconds')
        if return_nearest_neigbours_results:
            return test_results, {'knn_indeises': hole_test_data_indesies, 'knn_distance': hole_test_data_distances}
        return test_results

    def predict(self, test_data):
        return self(test_data)
