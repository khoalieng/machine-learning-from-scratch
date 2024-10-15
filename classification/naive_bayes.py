import math


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self._classes = list(set(y))
        n_classes = len(self._classes)

        # Calculate mean, variance, and prior for each class
        self._means = [[0 for _ in range(n_features)] for _ in range(n_classes)]
        self._vars = [[0 for _ in range(n_features)] for _ in range(n_classes)]
        self._priors = [0] * n_classes

        for cls_idx, cls in enumerate(self._classes):
            X_cls = [x_i for x_i, y_i in zip(X, y) if y_i == cls]
            n_cls_samples = len(X_cls)
            self._priors[cls_idx] = n_cls_samples / n_samples
            for feat_idx in range(n_features):
                self._means[cls_idx][feat_idx] = (
                    sum(x[feat_idx] for x in X_cls) / n_cls_samples
                )
                self._vars[cls_idx][feat_idx] = (
                    sum(
                        (x[feat_idx] - self._means[cls_idx][feat_idx]) ** 2
                        for x in X_cls
                    )
                    / n_cls_samples
                )
        return self

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        log_posteriors = []

        # Calculate posterior probability for each class
        for cls_idx in range(len(self._classes)):
            # Using log to avoid numerical underflow
            log_prior = math.log(self._priors[cls_idx])
            epsilon = 1e-9  # for numerical stability of log
            log_likelihood = sum(
                math.log(max(self._pdf(cls_idx, feat_idx, x[feat_idx]), epsilon))
                for feat_idx in range(len(x))
            )
            posterior = log_likelihood + log_prior
            log_posteriors.append(posterior)

        max_posterior_class = self._classes[log_posteriors.index(max(log_posteriors))]
        return max_posterior_class

    def _pdf(self, cls_idx, feat_idx, x_ij):
        mean = self._means[cls_idx][feat_idx]
        var = self._vars[cls_idx][feat_idx]
        # Gaussian PDF
        numerator = math.exp(-((x_ij - mean) ** 2) / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        return numerator / denominator
