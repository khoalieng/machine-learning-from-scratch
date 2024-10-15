class StandardScaler:
    def __init__(self):
        self.means = None
        self.std_devs = None

    def fit(self, X):
        self.n_samples, self.n_features = len(X), len(X[0])

        # Calculate mean for each feature
        self.means = []
        for j in range(self.n_features):
            sum_ = sum(x_i[j] for x_i in X)
            self.means.append(sum_ / self.n_samples)

        # Calculate standard deviation for each feature
        self.std_devs = []
        for j in range(self.n_features):
            variance_ = sum((x_i[j] - self.means[j]) ** 2 for x_i in X) / self.n_samples
            self.std_devs.append(variance_**0.5)
        return self

    def transform(self, X):
        X_transformed = []
        for x_i in X:
            transformed = [
                (x_i[j] - self.means[j]) / self.std_devs[j]
                for j in range(self.n_features)
            ]
            X_transformed.append(transformed)
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        X = []
        for x_i in X_transformed:
            original = [
                x_i[j] * self.std_devs[j] + self.means[j]
                for j in range(self.n_features)
            ]
            X.append(original)
        return X
