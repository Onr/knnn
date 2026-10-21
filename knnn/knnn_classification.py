"""Classification by fitting one k-NNN model per class."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .knnn_main import KNNN, validate_data


class KNNNClassifier(ClassifierMixin, BaseEstimator):
    """Predict the class with the smallest class-specific anomaly score."""

    def __init__(self, number_of_neighbors=3, number_of_neighbors_of_neighbors=25,
                 set_size=5, distance_function='cosine', p=2, batch_size=256):
        self.number_of_neighbors = number_of_neighbors
        self.number_of_neighbors_of_neighbors = number_of_neighbors_of_neighbors
        self.set_size = set_size
        self.distance_function = distance_function
        self.p = p
        self.batch_size = batch_size

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(y)
        classes, inverse = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError('At least two classes are required; got 1 class')
        if np.any(np.bincount(inverse) < 2):
            raise ValueError('Each class must have at least two training samples')
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.models_ = []
        for i in range(len(classes)):
            model = KNNN(
                number_of_neighbors=self.number_of_neighbors,
                number_of_neighbors_of_neighbors=self.number_of_neighbors_of_neighbors,
                set_size=self.set_size, distance_function=self.distance_function,
                p=self.p, batch_size=self.batch_size)
            self.models_.append(model.fit(X[inverse == i]))
        return self

    def _class_scores(self, X, return_neighbors=False):
        check_is_fitted(self, 'models_')
        X = validate_data(self, X, dtype=np.float64, reset=False)
        results = [model(X, return_nearest_neigbours_results=return_neighbors)
                   for model in self.models_]
        if return_neighbors:
            return (np.column_stack([result[0] for result in results]),
                    np.column_stack([result[1]['knn_distance'].mean(axis=1)
                                     for result in results]))
        return np.column_stack(results)

    def predict(self, X):
        scores = self._class_scores(X)
        return self.classes_[scores.argmin(axis=1)]

    def predict_proba(self, X):
        """Return votes, splitting ties equally; these are not calibrated probabilities."""
        return _minimum_votes(self._class_scores(X))


def _minimum_votes(scores):
    winners = scores == scores.min(axis=1, keepdims=True)
    return winners / winners.sum(axis=1, keepdims=True)


class KNNN_class(KNNNClassifier):
    """Compatibility interface retaining the original callable tuple results.

    New code should use KNNNClassifier. ``number_of_classes`` is checked at fit
    time and never changed, so cloning and refitting preserve constructor values.
    """

    def __init__(self, number_of_classes, number_of_neighbors=3,
                 number_of_neighbors_of_neighbors=25, set_size=5,
                 distance_function='cosine', p=2, batch_size=256):
        self.number_of_classes = number_of_classes
        super().__init__(number_of_neighbors, number_of_neighbors_of_neighbors,
                         set_size, distance_function, p, batch_size)

    def fit(self, embedding, gt=None, *, y=None):
        if gt is not None and y is not None:
            raise ValueError('Pass either gt or y, not both')
        labels = gt if gt is not None else y
        if labels is None:
            raise ValueError('Class labels are required')
        if len(np.unique(labels)) != self.number_of_classes:
            raise ValueError('number_of_classes must match the training labels')
        super().fit(embedding, labels)
        self.num_of_classes = len(self.classes_)
        self.class_to_index = {label: i for i, label in enumerate(self.classes_)}
        self.knnn_by_class = self.models_
        return self

    def __call__(self, test_embedding, return_nearest_neigbours_results=True):
        if return_nearest_neigbours_results:
            scores, neighbor_scores = self._class_scores(test_embedding, True)
            return (scores, neighbor_scores, _minimum_votes(scores),
                    _minimum_votes(neighbor_scores))
        scores = self._class_scores(test_embedding)
        return scores, _minimum_votes(scores)
