# %%
import numpy as np


class Elasticregression:
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.001,
        alpha: float = 0.2,
        beta: float = 0.1,
    ):
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.w = None
        self.b = 0.0

    def predict(self, x: np.ndarray):
        return np.dot(x, self.w) + self.b

    def update_weights(self, x: np.ndarray, y: np.ndarray, y_preds: np.ndarray):
        n = x.shape[0]
        self.w -= self.lr * (
            (-2 / n) * np.dot(x.T, (y - y_preds))
            + 2 * self.alpha * self.w
            + self.beta * np.sign(self.w)
        )
        self.b -= self.lr * -2 * np.mean(y - y_preds)

    def loss(self, y: np.ndarray, y_preds: np.ndarray):
        return np.sqrt(np.mean((y - y_preds) ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        y = np.reshape(y, (-1, 1))
        self.w = np.zeros((n_features, 1), dtype=float)
        losses = []

        for i in range(self.epochs):
            y_preds = self.predict(x)
            self.update_weights(x, y, y_preds)
            losses.append(self.loss(y, y_preds))

        return losses


# %%

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x, y = make_regression(n_samples=5000, n_features=10, shuffle=True, random_state=42)

x.shape, y.shape
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True)

# %%
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# %%
x_train[:10], y[:10]
# %%
er = Elasticregression(epochs=500, lr=0.01, alpha=0.01, beta=0.1)
losses = er.fit(x_train, y_train)

y_preds = er.predict(x_test)

print(f"train loss :{losses[-1]}")

print(f"test loss : {er.loss(y_preds.reshape(-1,1),y_test.reshape(-1,1))}")
# %%
print(y_test)
# %%
print(y_preds)
# %%
from sklearn.linear_model import ElasticNet

ers = ElasticNet()
ers.fit(x_train, y_train)
y_preds = ers.predict(x_test).reshape(-1, 1)
print(f"skelarn model loss : {er.loss(y_test.reshape(-1,1),y_preds)}")
# %%
print(y_preds)
# %%
