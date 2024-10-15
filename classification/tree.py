import random
import math

from collections import Counter


class Node:
    def __init__(
        self, feature_idx=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(
        self, min_samples_split=2, max_depth=100, max_features=None, random_state=None
    ):
        if random_state is not None:
            random.seed(random_state)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        n_available_features = len(X[0])
        self.max_features = (
            n_available_features
            if not self.max_features
            else min(n_available_features, self.max_features)
        )
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_available_features = len(X), len(X[0])
        n_labels = len(set(y))

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = random.sample(range(n_available_features), self.max_features)

        # Find the best split
        best_feature_idx, best_threshold = self._best_split(X, y, feature_indices)

        # Create children nodes recursively
        left_indices, right_indices = self._split(
            [row[best_feature_idx] for row in X], best_threshold
        )

        # Check if the split is valid (i.e., neither side is empty)
        if len(left_indices) == 0 or len(right_indices) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_child = self._grow_tree(
            [X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1
        )
        right_child = self._grow_tree(
            [X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1
        )
        return Node(best_feature_idx, best_threshold, left_child, right_child)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_indices:
            feature = [row[feature_idx] for row in X]
            thresholds = set(feature)

            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, feature, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_indices, right_indices = self._split(feature, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Calculate the weighted average entropy of children
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        e_left, e_right = (
            self._entropy([y[i] for i in left_indices]),
            self._entropy([y[i] for i in right_indices]),
        )
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, feature, split_threshold):
        left_indices = [
            idx for idx, val in enumerate(feature) if val <= split_threshold
        ]
        right_indices = [
            idx for idx, val in enumerate(feature) if val > split_threshold
        ]
        return left_indices, right_indices

    def _entropy(self, y):
        counts, total = Counter(y), len(y)
        P = [val / total for val in counts.values()]
        return -sum(p * math.log(p) for p in P)

    def _most_common_label(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
