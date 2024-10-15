def unit_step(x):
    return 1 if x >= 0 else 0


class Perceptron:
    def __init__(self, lr=0.001, n_iters=1_000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = len(X[0])

        # Initialize weights and bias
        self.weights = [0] * n_features
        self.bias = 0

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear = sum(x_ij * w for x_ij, w in zip(x_i, self.weights)) + self.bias
                # Traditional perceptron (unit step activation function)
                y_pred = unit_step(linear)
                # Perceptron update rule (not gradient descent)
                update = self.lr * (y[idx] - y_pred)
                self.weights = [w + update * x_ij for w, x_ij in zip(self.weights, x_i)]
                self.bias += update
        return self

    def predict(self, X):
        y_pred = []
        for x_i in X:
            linear = sum(x_ij * w for x_ij, w in zip(x_i, self.weights)) + self.bias
            y_pred.append(unit_step(linear))
        return y_pred
