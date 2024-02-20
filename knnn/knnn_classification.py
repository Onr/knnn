from typing import List, Any

import numpy as np

from knnn import KNNN

class KNNN_class(KNNN):
    """
    Get KNNN results for classification task
    """
    def __init__(self, number_of_classes: int, **kwargs) -> None:
        # check that args do not contain embedding and gt
        assert kwargs.keys().__contains__("embedding") is False, "embedding must not be in args for KNNN_class"
        assert kwargs.keys().__contains__("gt") is False, "gt must not be in args for KNNN_class"
        
        self.num_of_classes = number_of_classes
        self.knnn_by_class = []
        for _ in range(number_of_classes):
            self.knnn_by_class.append(KNNN(**kwargs))
    

    def fit(self, embedding: np.ndarray, gt: List[any]) -> None:
        self.gt = gt
        # convert class to index
        self.class_to_index = {c: i for i, c in enumerate(set(gt))}
        self.number_of_classes = self.class_to_index.__len__()
        self.embedding_by_class = [embedding[gt == c] for c in self.class_to_index.keys()]

        assert embedding.shape[0] == self.gt.__len__(), "embedding and gt must have the same length"
        assert self.number_of_classes > 1, "number of classes must be greater than 1"
        assert self.number_of_classes < self.gt.__len__(), "number of classes must be less than number of samples"

        # fit knnn for each class
        for i, cur_knnn in enumerate(self.knnn_by_class):
            cur_knnn.fit(self.embedding_by_class[i])
        

    def __call__(self, test_embedding: np.ndarray, return_nearest_neigbours_results: bool = True) -> List[float]:

        # if self.first_global_normalization:
        #     test_embedding, _, _ = embedding_whitening(test_embedding, with_means=self.global_means, with_stds=self.global_stds)   
        # get knnn results for each class
        knnn_results_by_class = []
        knn_results_by_class = []
        for i, cur_knnn in enumerate(self.knnn_by_class):
            if return_nearest_neigbours_results:
                knnn_results, knn_results_dict = cur_knnn(test_embedding, return_nearest_neigbours_results=return_nearest_neigbours_results)
                knn_results = knn_results_dict['knn_distance'].mean(1)
                knn_results_by_class.append(knn_results)
            else:
                knnn_results = cur_knnn(test_embedding, return_nearest_neigbours_results=return_nearest_neigbours_results)
            knnn_results_by_class.append(knnn_results)

        if return_nearest_neigbours_results:
            knn_results_by_class = np.stack(knn_results_by_class, axis=1)
        knnn_results_by_class = np.stack(knnn_results_by_class, axis=1)

        # convert the anomaly score to a Classification score
        knnn_results_by_class_bin = np.zeros_like(knnn_results_by_class)
        for row_ind in range(knnn_results_by_class.shape[0]):
            min_inds = np.where(knnn_results_by_class[row_ind] == np.min(knnn_results_by_class[row_ind]))[0]
            knnn_results_by_class_bin[row_ind, min_inds] = 1 / min_inds.__len__()

        if return_nearest_neigbours_results:
            knn_results_by_class_bin = np.zeros_like(knn_results_by_class)
            for row_ind in range(knn_results_by_class.shape[0]):
                min_inds = np.where(knn_results_by_class[row_ind] == np.min(knn_results_by_class[row_ind]))[0]
                knn_results_by_class_bin[row_ind, min_inds] = 1 / min_inds.__len__()

        if return_nearest_neigbours_results:
            return knnn_results_by_class, knn_results_by_class, knnn_results_by_class_bin, knn_results_by_class_bin
        return knnn_results_by_class, knnn_results_by_class_bin

    
