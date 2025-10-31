import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))

        # Sort by eigenvalues (descending)
        idxs = np.argsort(abs(eig_vals))[::-1]
        self.W = eig_vecs[:, idxs[: self.n_components]].real

    def transform(self, X):
        return np.dot(X, self.W)


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

LDA = LDA(2)
LDA.fit(X, y)
X_reduced = LDA.transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
