import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(42)


class KMeansScratch:
    def __init__(self, n_clusters=3, epochs=10, tol=1e-6):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.tol = tol
        self.centroids = None

    def update_centroids(self, x):
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(x[self.labels == i], axis=0)

    def fit(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        self.centroids = np.random.normal(
            loc=mean, scale=std, size=(self.n_clusters, x.shape[1])
        )
        n_samples = x.shape[0]
        self.labels = np.full(shape=n_samples, fill_value=-1)
        old_centroids = self.centroids.copy()

        for _ in range(self.epochs):
            for idx, point in enumerate(x):
                distances = cdist(point.reshape(1, -1), self.centroids)
                label = np.argmin(distances.flatten())
                self.labels[idx] = label
            self.update_centroids(x)

            change = np.sum(np.abs(self.centroids - old_centroids) > self.tol)
            if change < self.n_clusters / 2:
                break

            old_centroids = self.centroids.copy()

        return self

    def predict(self, x):
        distances = cdist(x, self.centroids)
        labels = np.argmin(distances, axis=1)
        return labels


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

kmeans = KMeansScratch(n_clusters=3)
kmeans.fit(X)
preds = kmeans.predict(X)

print("Cluster centroids:\n", kmeans.centroids)
print("First  predicted clusters:", preds)

from matplotlib import pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap="rainbow")
plt.title("kmeans from Scratch")
plt.show()
