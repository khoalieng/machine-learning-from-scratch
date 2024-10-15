import random


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    if random_state is not None:
        random.seed(random_state)

    assert len(X) == len(y), "The length of X and y must be equal."

    if shuffle:
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
