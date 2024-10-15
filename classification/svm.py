class SVM:
    def __init__(self, lr=0.001, lmbda=0.01, n_iters=1_000):
        self.lr = lr
        self.lmbda = lmbda
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = len(X[0])

        # Initialize weights and bias
        self.weights = [0] * n_features
        self.bias = 0

        # y needs to be either -1 or 1
        y_ = [-1 if val == 0 else 1 for val in y]

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                # Update rule
                condition = (
                    y_[i]
                    * (
                        sum(x_i[j] * self.weights[j] for j in range(n_features))
                        - self.bias
                    )
                    >= 1
                )
                if condition:
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * self.lmbda * self.weights[j])
                else:
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (
                            2 * self.lmbda * self.weights[j] - y_[i] * x_i[j]
                        )
                    self.bias -= self.lr * y_[i]
        return self

    def predict(self, X):
        return [1 if self._predict(x) >= 1 else 0 for x in X]

    def _predict(self, x_i):
        linear = sum(x_i[j] * self.weights[j] for j in range(len(x_i))) - self.bias
        return linear
