from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # sklearn < 1.6
    def validate_data(estimator, *args, **kwargs):
        return estimator._validate_data(*args, **kwargs)


EPSILON = 1e-10


def compute_correlation_matrix(data):
    """
    Computes the correlation matrix for a given 2D NumPy array.
    
    :param data: A 2D NumPy array of size [n, m] where n is the number of samples
                 and m is the number of features.
    :return: The correlation matrix of size [m, m].
    """
    data = check_array(data, dtype=np.float64, ensure_min_samples=2)
    centered = data - data.mean(axis=0)
    scales = np.linalg.norm(centered, axis=0)
    standardized_data = np.divide(
        centered, scales, out=np.zeros_like(centered), where=scales != 0
    )

    # Compute the correlation matrix
    correlation_matrix = np.nan_to_num(standardized_data.T @ standardized_data)
    correlation_matrix = np.clip(correlation_matrix, -1, 1)

    return correlation_matrix

    
def hierarchical_clustering(correlation_matrix, do_plot=False):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

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
    
    from scipy.cluster.hierarchy import fcluster

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
    Group correlated features into blocks of at most target_size entries.
    
    :param data: Data matrix with features in columns.
    :param target_size: Desired number of features in each cluster.
    :return: Array indicating cluster membership for each feature.
    """
    order = reorder_features_by_correlation(data, target_size)
    clusters = np.empty(data.shape[1], dtype=int)
    clusters[order] = np.arange(data.shape[1]) // target_size
    return clusters


def reorder_features_by_correlation(data, L):
    """Return a feature permutation using the greedy rule in Section 3.

    Within a block, choose the remaining feature most correlated (in absolute
    value) with the preceding two entries. Start each new block with the least
    correlated remaining feature. Constant features have zero correlation.
    """
    if not isinstance(L, Integral) or isinstance(L, bool) or L < 1:
        raise ValueError("L must be a positive integer")
    correlations = np.abs(compute_correlation_matrix(data))
    constant = np.flatnonzero(np.ptp(data, axis=0) == 0)
    order = np.flatnonzero(np.ptp(data, axis=0) != 0)
    for i in range(1, len(order)):
        block_start = (i // L) * L
        start = max(block_start, i - 2) if i % L else max(0, i - 2)
        remaining = order[i:]
        values = correlations[np.ix_(order[start:i], remaining)].mean(axis=0)
        offset = np.argmin(values) if i % L == 0 else np.argmax(values)
        selected = i + offset
        order[i], order[selected] = order[selected], order[i]
    # Constant columns carry no grouping information and should not split
    # correlated pairs simply by winning a zero-correlation boundary tie.
    return np.concatenate([order, constant])


def get_features_sets(data, clusters):
    sets = []
    for cluster in np.unique(clusters):
        sets.append(np.ascontiguousarray(data[:, clusters == cluster]))
    return sets


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms != 0)


def get_features_knn_faiss(data_1, data_2=None, dist_func='cosine', k=25, do_gpu=False, index=None, return_index=False, return_centers=False):
    # Retained for callers of the legacy helper; the estimator uses sklearn.
    import faiss

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
    
    for i in range(n):
        # Select the embedding of the sample and its nearest neighbors
        neighbors = indices_matrix[i]
        sub_matrix = embedding_matrix[neighbors, :]

        # compute the center of the neighbors
        centers[i] = sub_matrix[0, :] if not use_means_as_return__centers else np.mean(sub_matrix, axis=0) 
        
        # Compute the covariance matrix for the sample and its neighbors
        cov_matrix = np.atleast_2d(np.cov(sub_matrix, rowvar=False))

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

        
def calc_eigen_for_set(data, indices, batch_size=256):
    """Compute local covariance eigenvectors and standard deviations in batches.

    The neighborhood includes the training point itself. Keeping covariance
    matrices three dimensional also handles a final block of just one feature.
    """
    n, dimensions = data.shape
    eigen_values = np.empty((n, dimensions))
    eigen_vectors = np.empty((n, dimensions, dimensions))
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        neighbors = data[indices[start:stop]]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = centered.transpose(0, 2, 1) @ centered
        covariance /= indices.shape[1] - 1
        values, vectors = np.linalg.eigh(covariance)
        eigen_values[start:stop] = np.sqrt(np.maximum(values, 0))
        eigen_vectors[start:stop] = vectors
    return eigen_values, eigen_vectors


class KNNN(BaseEstimator):
    """Nearest neighbors of neighbors anomaly scores (larger is more anomalous).

    ``set_size`` is the maximum feature block size; -1 uses all features.
    ``p=1`` implements the paper's absolute projections divided by local
    standard deviations. Scores are averaged over neighbors and dimensions.
    The default ``p=2`` uses RMS Mahalanobis distances in each block, with
    Euclidean distances for singular covariances as in the original estimator.
    ``batch_size`` bounds the temporary arrays used in fitting and inference.

    For compatibility, ``predict`` and ``__call__`` return continuous anomaly
    scores. ``score_samples`` follows sklearn's opposite sign convention.
    """

    def __init__(self, number_of_neighbors=3, number_of_neighbors_of_neighbors=25,
                 set_size=5, distance_function='cosine', p=2, batch_size=256):
        self.number_of_neighbors = number_of_neighbors
        self.number_of_neighbors_of_neighbors = number_of_neighbors_of_neighbors
        self.set_size = set_size
        self.distance_function = distance_function
        self.p = p
        self.batch_size = batch_size

    @property
    def data(self):
        return self.data_

    @property
    def features_clusters(self):
        return self.features_clusters_

    @property
    def feature_sets(self):
        return self.feature_sets_

    @property
    def fitted(self):
        return getattr(self, 'fitted_', False)

    def fit(self, data, y=None):
        data = validate_data(self, data, dtype=np.float64,
                             ensure_min_samples=2, copy=True)
        for name in ('number_of_neighbors', 'number_of_neighbors_of_neighbors',
                     'batch_size', 'set_size'):
            value = getattr(self, name)
            if (not isinstance(value, Integral) or isinstance(value, bool)
                    or (value < 1 and not (name == 'set_size' and value == -1))):
                raise ValueError(f"{name} must be a positive integer"
                                 + (" or -1" if name == 'set_size' else ""))
        if self.number_of_neighbors_of_neighbors < 2:
            raise ValueError("number_of_neighbors_of_neighbors must be at least 2")
        if self.distance_function not in ('cosine', 'euclidean'):
            raise ValueError("distance_function must be 'cosine' or 'euclidean'")
        if self.p not in (1, 2):
            raise ValueError("p must be 1 or 2")

        self.n_features_in_ = data.shape[1]
        self.data_ = data
        self.n_neighbors_ = min(self.number_of_neighbors, len(data))
        self.n_eigen_neighbors_ = min(self.number_of_neighbors_of_neighbors, len(data))
        self.set_size_ = (self.n_features_in_ if self.set_size == -1
                          else min(self.set_size, self.n_features_in_))
        self.neighbors_ = NearestNeighbors(metric=self.distance_function).fit(data)
        self.features_clusters_ = get_features_clusters(data, self.set_size_)
        self.feature_indices_ = [np.flatnonzero(self.features_clusters_ == group)
                                 for group in np.unique(self.features_clusters_)]
        self.feature_sets_ = [data[:, features] for features in self.feature_indices_]
        self.constant_features_ = np.flatnonzero(np.ptp(data, axis=0) == 0)
        self.constant_values_ = data[0, self.constant_features_].copy()
        self.eigen_values_groups_ = []
        self.eigen_vectors_groups_ = []
        # Query neighborhoods in chunks too, rather than allocating N x N
        # cosine distances for a large training set.
        indices = np.empty((len(data), self.n_eigen_neighbors_), dtype=int)
        for start in range(0, len(data), self.batch_size):
            stop = start + self.batch_size
            indices[start:stop] = self.neighbors_.kneighbors(
                data[start:stop], n_neighbors=self.n_eigen_neighbors_,
                return_distance=False)
        for feature_set in self.feature_sets_:
            values, vectors = calc_eigen_for_set(feature_set, indices,
                                                 self.batch_size)
            self.eigen_values_groups_.append(values)
            self.eigen_vectors_groups_.append(vectors)
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, 'fitted_', False)

    def __call__(self, test_data, return_nearest_neigbours_results=False):
        check_is_fitted(self)
        test_data = validate_data(self, test_data, dtype=np.float64, reset=False)
        scores = np.zeros(len(test_data))
        if return_nearest_neigbours_results:
            all_distances = np.empty((len(test_data), self.n_neighbors_))
            all_indices = np.empty((len(test_data), self.n_neighbors_), dtype=int)
        for start in range(0, len(test_data), self.batch_size):
            stop = min(start + self.batch_size, len(test_data))
            queries = test_data[start:stop]
            distances, indices = self.neighbors_.kneighbors(
                queries, n_neighbors=self.n_neighbors_)
            for features, feature_set, values, vectors in zip(
                    self.feature_indices_, self.feature_sets_, self.eigen_values_groups_,
                    self.eigen_vectors_groups_):
                differences = queries[:, None, features] - feature_set[indices]
                projections = np.einsum('nkd,nkdl->nkl', differences,
                                        vectors[indices], optimize=True)
                scales = values[indices] + EPSILON
                if self.p == 2:
                    # The original estimator used an identity inverse when
                    # covariance inversion failed. Detect numerical singularity
                    # from the eigenvalues to avoid unstable inverse matrices.
                    local_values = values[indices]
                    singular = (local_values[..., 0] ** 2 <= np.finfo(float).eps
                                * len(features) * local_values[..., -1] ** 2)
                    scales = np.where(singular[..., None], 1.0, scales)
                standardized = np.abs(projections) / scales
                if self.p == 1:
                    block_scores = standardized.mean(axis=2)
                else:
                    block_scores = np.sqrt(np.mean(standardized ** 2, axis=2))
                scores[start:stop] += (block_scores.mean(axis=1)
                                       * len(features) / self.n_features_in_)
            changed_constants = np.any(
                queries[:, self.constant_features_] != self.constant_values_, axis=1)
            scores[start:stop][changed_constants] = np.finfo(np.float64).max
            if return_nearest_neigbours_results:
                # FAISS returned squared Euclidean distances in the old API.
                all_distances[start:stop] = (distances ** 2
                                             if self.distance_function == 'euclidean'
                                             else distances)
                all_indices[start:stop] = indices
        if return_nearest_neigbours_results:
            return scores, {'knn_indeises': all_indices, 'knn_distance': all_distances}
        return scores

    def predict(self, test_data):
        """Return continuous anomaly scores; larger values are more anomalous."""
        return self(test_data)

    def score_samples(self, X):
        """Return normality scores, with larger values indicating normal samples."""
        return -self(X)
