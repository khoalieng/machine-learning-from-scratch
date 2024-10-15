import random


def euclidean_distance(x1, x2):
    return sum((x1_j - x2_j) ** 2 for x1_j, x2_j in zip(x1, x2)) ** 0.5


class KMeans:
    def __init__(
        self,
        k=3,
        max_iters=100,
        random_state=None,
    ):
        if random_state:
            random.seed(random_state)
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X, y=None):
        self.n_samples, self.n_features = len(X), len(X[0])

        # Initialize clusters
        random_indices = random.sample(range(self.n_samples), self.k)
        self.centroids = [X[i] for i in random_indices]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to their closest centroid
            clusters = self._create_clusters(X, self.centroids)
            prev_centroids = self.centroids
            self.centroids = self._get_centroids(X, clusters)
            if self._is_converged(prev_centroids, self.centroids):
                break
        return self

    def predict(self, X):
        clusters = self._create_clusters(X, self.centroids)
        labels = [self._closest_centroid(x, self.centroids) for x in X]
        return labels, clusters, self.centroids

    def _create_clusters(self, X, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.k)]
        for i, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(i)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = distances.index(min(distances))
        return closest_idx

    def _get_centroids(self, X, clusters):
        # Assign mean value of cluster to centroids
        centroids = [[0 for _ in range(self.n_features)] for _ in range(self.k)]
        for cluster_idx, cluster in enumerate(clusters):
            for i in cluster:
                for j in range(self.n_features):
                    centroids[cluster_idx][j] += X[i][j]  # sum
            centroids[cluster_idx] = [
                feat / len(cluster) for feat in centroids[cluster_idx]
            ]  # average
        return centroids

    def _is_converged(self, prev_centroids, curr_centroids):
        # Distances between previous and current centroid, for all centroids
        distances = [
            euclidean_distance(prev_centroids[i], curr_centroids[i])
            for i in range(self.k)
        ]
        return sum(distances) == 0
