import random

from collections import Counter
from classification.tree import DecisionTree


class Bagging:
    def __init__(
        self,
        n_estimators=10,
        base_estimator=None,
        base_estimator_params=None,
        random_state=None,
    ):
        if random_state is not None:
            random.seed(random_state)
        self.base_estimator = base_estimator if base_estimator else DecisionTree
        self.base_estimator_params = (
            base_estimator_params if base_estimator_params else {}
        )
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y)
            estimator = self.base_estimator(**self.base_estimator_params)
            self.estimators.append(estimator.fit(X_sample, y_sample))
        return self

    def predict(self, X):
        all_preds = [estimator.predict(X) for estimator in self.estimators]
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
