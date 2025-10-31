import numpy as np
from scipy.spatial.distance import cdist


class AgglomerativeClusteringScratch:
    def __init__(self, n_clusters=2, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        clusters = [[i] for i in range(len(X))]
        distances = cdist(X, X)

        np.fill_diagonal(distances, np.inf)
        while len(clusters) > self.n_clusters:

            i, j = np.unravel_index(np.argmin(distances), distances.shape)

            new_cluster = clusters[i] + clusters[j]
            clusters[i] = new_cluster
            clusters.pop(j)

            distances = self._update_distances(X, clusters)

        self.clusters = clusters
        return self

    def _update_distances(self, X, clusters):
        n = len(clusters)
        new_dist = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d = cdist(X[clusters[i]], X[clusters[j]]).mean()
                new_dist[i, j] = new_dist[j, i] = d
        return new_dist

    def get_labels(self):
        labels = np.zeros(sum(len(c) for c in self.clusters), dtype=int)
        for i, cluster in enumerate(self.clusters):
            for idx in cluster:
                labels[idx] = i
        return labels


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=3, random_state=42)


hc = AgglomerativeClusteringScratch(n_clusters=3, linkage="ward")
hc.fit(X)
labels = hc.get_labels()

labels
