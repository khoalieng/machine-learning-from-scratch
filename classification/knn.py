from collections import Counter


def euclidean_distance(x1, x2):
    return sum((x1_j - x2_j) ** 2 for x1_j, x2_j in zip(x1, x2)) ** 0.5


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the k closest neighbors
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        return self._most_common_label(k_nearest_labels)

    def _most_common_label(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label
