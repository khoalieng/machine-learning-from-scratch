def precision_score(y_true, y_pred):
    true_positives = sum(true == 1 and pred == 1 for true, pred in zip(y_true, y_pred))
    predicted_positives = sum(pred == 1 for pred in y_pred)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall_score(y_true, y_pred):
    true_positives = sum(true == 1 and pred == 1 for true, pred in zip(y_true, y_pred))
    actual_positives = sum(true == 1 for true in y_true)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def accuracy_score(y_true, y_pred):
    total = len(y_true)
    if total <= 0:
        return 0.0
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / total


def mean_squared_error(y_true, y_pred):
    total = len(y_true)
    if total <= 0:
        return 0.0
    sum_squared_error = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    return sum_squared_error / total


def mean_absolute_error(y_true, y_pred):
    total = len(y_true)
    if total <= 0:
        return 0.0
    sum_absolute_error = sum(abs(true - pred) for true, pred in zip(y_true, y_pred))
    return sum_absolute_error / total
