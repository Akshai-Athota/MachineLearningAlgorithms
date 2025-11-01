# %%
import numpy as np
from scipy.spatial.distance import cdist
from collections import deque


class DBSCAN_OWN:
    def __init__(self, eps: float = 1.5, min_points: int = 5):
        self.e = eps
        self.min_points = min_points

    def fit(self, x: np.ndarray):
        n_samples = x.shape[0]
        distances = cdist(x, x)
        np.fill_diagonal(distances, np.inf)
        self.visited = np.zeros(n_samples, dtype=bool)

        self.labels = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            if self.visited[i]:
                continue

            self.visited[i] = True
            neighbours = np.where(distances[i] <= self.e)[0]

            if neighbours.shape[0] < self.min_points:
                self.labels[i] = -1
            else:
                self._expand_cluster(neighbours, distances, cluster_id)
                self.labels[i] = cluster_id
                cluster_id += 1

        self.n_clusters = cluster_id

        return self

    def _expand_cluster(self, neighbours, distances, cluster_id):
        queue = deque(neighbours)

        while queue:
            point = queue.popleft()

            if self.visited[point] == False:
                self.visited[point] = True
                new_neighbours = np.where(distances[point] <= self.e)[0]
                if new_neighbours.shape[0] >= self.min_points:
                    queue.extend(new_neighbours)

                if self.labels[point] == -1:
                    self.labels[point] = cluster_id

    def get_labels(self):
        return self.labels


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

db = DBSCAN_OWN(eps=0.2, min_points=5)
db.fit(X)
labels = db.get_labels()

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.title("DBSCAN from Scratch")
plt.show()
print(labels)
# %%
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=0.2, min_samples=5)
dbs.fit(X)
labels = dbs.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.title("DBSCAN from Scratch")
plt.show()
print(labels)
# %%
