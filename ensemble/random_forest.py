import random

from collections import Counter
from classification.tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, tree_params=None, random_state=None):
        if random_state is not None:
            random.seed(random_state)
        self.n_trees = n_trees
        self.tree_params = tree_params if tree_params else {}

    def fit(self, X, y):
        self.trees = []
        # Use a random subset of the available features per tree
        self.tree_params["max_features"] = self.tree_params.get(
            "max_features", int(len(X[0]) ** 0.5)
        )
        for _ in range(self.n_trees):
            tree = DecisionTree(**self.tree_params)
            X_sample, y_sample = self._bootstrap(X, y)
            self.trees.append(tree.fit(X_sample, y_sample))
        return self

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        all_preds_transposed = [
            [preds[j] for preds in all_preds] for j in range(len(all_preds[0]))
        ]
        return self._majority_vote(all_preds_transposed)

    def _majority_vote(self, all_preds_transposed):
        y_pred = []
        for preds in all_preds_transposed:
            most_common_pred = Counter(preds).most_common(1)[0][0]
            y_pred.append(most_common_pred)
        return y_pred

    def _bootstrap(self, X, y):
        n_samples = len(X)
        indices = random.choices(range(n_samples), k=n_samples)
        return [X[i] for i in indices], [y[i] for i in indices]
