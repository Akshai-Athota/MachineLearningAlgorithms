import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
rng = np.random.RandomState(42)

outliers = rng.uniform(low=-6, high=6, size=(20, 2))
X = np.concatenate([X, outliers], axis=0)

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
y_pred = iso.fit_predict(X)


X_normal = X[y_pred == 1]
X_anomaly = X[y_pred == -1]


plt.figure(figsize=(7, 6))
plt.scatter(X_normal[:, 0], X_normal[:, 1], c="blue", label="Normal")
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], c="red", label="Anomaly")
plt.legend()
plt.title("Isolation Forest — Outlier Detection")
plt.show()
