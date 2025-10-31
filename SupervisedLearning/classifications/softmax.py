# %%

import numpy as np


class Softmax:
    def __init__(self, epochs: int = 100, lr: float = 0.001):
        self.epochs = epochs
        self.lr = lr
        self.w = None

    def softmax(self, z):
        z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def one_hot(self, y: np.ndarray):
        n_classes = len(np.unique(y))
        y_hot = np.zeros((y.size, n_classes))
        y_hot[np.arange(y.size), y] = 1
        return y_hot

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        n_classes = len(np.unique(y))
        self.w = np.zeros((n_features, n_classes))
        y_hot = self.one_hot(y)

        for i in range(self.epochs):
            scores = x @ self.w
            probs = self.softmax(scores)
            grad = x.T @ (probs - y_hot) / n_samples
            self.w -= self.lr * grad

    def predict_proba(self, X):
        return self.softmax(X @ self.W)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Sklearn Softmax Regression Accuracy:", accuracy)
# %%
