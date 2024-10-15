import csv
import matplotlib.pyplot as plt


def convert_value(value):
    if value.isdigit() or (value[0] == "-" and value[1:].isdigit()):
        return int(value)
    else:
        try:
            return float(value)
        except ValueError:
            return value


def load_data(filename):
    with open(f"data/{filename}", newline="") as csvfile:
        data = list(csv.reader(csvfile))
    features = [[convert_value(val) for val in row[:-1]] for row in data[1:]]
    target = [convert_value(row[-1]) for row in data[1:]]
    return {"features": features, "target": target}


def kmeans_plot(X_2d, clusters, centroids):
    _, ax = plt.subplots(figsize=(8, 5))
    for cluster in clusters:
        points = [[X_2d[i][j] for i in cluster] for j in range(len(X_2d[0]))]
        ax.scatter(points[0], points[1])
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], marker="+", color="black", linewidth=2)
    plt.show()
