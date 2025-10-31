# %%
import numpy as np

np.random.seed(42)


class Precepton:
    def __init__(self, epochs: int = 100, lr: float = 0.001):
        self.epochs = epochs
        self.lr = lr
        self.w = None
        self.b = 0

    def accuracy(self, y_preds: np.ndarray, y: np.ndarray):
        z = y == y_preds
        return (z.sum() / y.shape[0]) * 100

    def predict(self, x: np.ndarray):
        z = np.dot(x, self.w) + self.b
        return np.sign(z)

    def actual_preds(self, x: np.ndarray):
        preds = self.predict(x)
        preds[preds == -1] = 0
        return preds

    def update_weights(self, y_preds: np.ndarray, x: np.ndarray, y: np.ndarray):
        conditions = y * y_preds
        for idx, condition in enumerate(conditions):
            if condition < 0.0:
                self.w += self.lr * ((y[idx] * x[idx]).reshape(-1, 1))
                self.b += self.lr * y[idx]

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_featurres = x.shape
        self.w = np.ones((n_featurres, 1), dtype=float)
        accuracy = []
        y[y == 0] = -1
        y = np.reshape(y, (-1, 1))
        for i in range(self.epochs):
            y_preds = self.predict(x)
            self.update_weights(y_preds, x, y)
            accuracy.append(self.accuracy(y, y_preds))

        return accuracy


# %%

from sklearn.datasets import make_classification

x, y = make_classification(
    n_samples=10000,
    n_features=6,
    n_informative=2,
    random_state=42,
    n_redundant=0,
    return_X_y=True,
)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

pc = Precepton(epochs=350, lr=0.0001)
pc.fit(x_train, y_train)
y_preds = pc.actual_preds(x_test)

y_preds.shape, y_test.shape

# %%
print(f"accuracy of own model:{pc.accuracy(y_preds,y_test.reshape((-1,1)))}")

# %%
y_preds, y_test

# %%
