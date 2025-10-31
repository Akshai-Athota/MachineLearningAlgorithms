# %%
import numpy as np

np.random.seed(42)


class Adaline:

    def __init__(self, epochs: int = 100, lr: float = 0.001):
        self.epochs = epochs
        self.lr = lr
        self.w = None
        self.b = 0.0

    def predict(self, x: np.ndarray):
        return np.dot(x, self.w) + self.b

    def accuracy(self, y: np.ndarray, y_preds: np.ndarray):
        z = y == y_preds
        return (z.sum() / y.shape[0]) * 100

    def update_weights(self, x: np.ndarray, y: np.ndarray, y_preds: np.ndarray):
        n = x.shape[0]
        self.w += self.lr * (1 / n) * (np.dot(x.T, y - y_preds))
        self.b += self.lr * np.mean(y - y_preds)

    def actual_preds(self, x: np.ndarray):
        preds = self.predict(x)
        z = np.sign(preds)
        z[z == -1] = 0
        return z

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.w = np.zeros((n_features, 1), dtype=float)
        y[y == 0] = -1
        y = y.reshape((-1, 1))
        accuracy = []

        for i in range(self.epochs):
            y_preds = self.predict(x)
            self.update_weights(x, y, y_preds)
            y_preds[y_preds < 0] = -1
            accuracy.append(self.accuracy(y_preds, y))

        return accuracy


from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)
adaline = Adaline(epochs=500, lr=0.00008)
acc_hist = adaline.fit(X, y)

print("Final accuracy:", acc_hist[-1])
print("Weights:", adaline.w.ravel(), "Bias:", adaline.b)

# %%
