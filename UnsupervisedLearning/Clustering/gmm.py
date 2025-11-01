# %%
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        np.random.seed(42)
        shuffled = np.random.permutation(X)
        self.mu = shuffled[: self.K]
        self.pi = np.ones(self.K) / self.K
        self.sigma = [np.cov(X.T) for _ in range(self.K)]

        log_likelihood_old = 0

        for iteration in range(self.max_iter):

            r = np.zeros((n_samples, self.K))
            for k in range(self.K):
                r[:, k] = self.pi[k] * multivariate_normal.pdf(
                    X, mean=self.mu[k], cov=self.sigma[k]
                )
            r = r / r.sum(axis=1, keepdims=True)

            N_k = r.sum(axis=0)
            self.mu = (r.T @ X) / N_k[:, np.newaxis]
            self.sigma = []
            for k in range(self.K):
                diff = X - self.mu[k]
                cov = (r[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
                self.sigma.append(cov)
            self.pi = N_k / n_samples

            log_likelihood = np.sum(
                np.log(
                    np.sum(
                        [
                            self.pi[k]
                            * multivariate_normal.pdf(X, self.mu[k], self.sigma[k])
                            for k in range(self.K)
                        ],
                        axis=0,
                    )
                )
            )
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        return self

    def predict(self, X):
        probs = np.array(
            [
                self.pi[k] * multivariate_normal.pdf(X, self.mu[k], self.sigma[k])
                for k in range(self.K)
            ]
        ).T
        return np.argmax(probs, axis=1)


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

gmm = GMM(n_components=3)
gmm.fit(X)
y_pred = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="rainbow")
plt.title("GMM Clustering (from scratch)")
plt.show()

from sklearn.mixture import GaussianMixture

gmm_sk = GaussianMixture(n_components=3)
gmm_sk.fit(X)
labels = gmm_sk.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.title("GMM using sklearn")
plt.show()

# %%
