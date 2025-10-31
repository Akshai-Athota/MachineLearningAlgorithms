# %%

import numpy as np


class pca:
    def __init__(self, n_features):
        self.features = n_features

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        x_centered = x - self.mean
        x_covar = np.cov(x_centered, rowvar=False)
        print(x_covar)

        eigen_values, eigen_vectors = np.linalg.eigh(x_covar)
        print(eigen_values)
        print(eigen_vectors)

        t = np.argsort(eigen_values)
        idx = t[::-1]
        self.v = eigen_vectors[:, idx][:, : self.features]
        print(self.v)

    def transform(self, x: np.ndarray):
        x = x - self.mean
        return np.dot(x, self.v)

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

pca = pca(2)
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
# %%
