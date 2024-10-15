class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1_000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0] * n_features
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            # Gradients calculation
            dw = [0] * n_features
            db = 0
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            # Average the gradients and update the weights and bias
            dw = [d / n_samples for d in dw]
            db /= n_samples
            self.weights = [w - self.lr * dw_j for w, dw_j in zip(self.weights, dw)]
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        prediction = sum(x_ij * w_j for x_ij, w_j in zip(x, self.weights)) + self.bias
        return prediction
