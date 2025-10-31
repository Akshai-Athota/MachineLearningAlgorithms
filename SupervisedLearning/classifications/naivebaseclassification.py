# %%

import numpy as np
from typing import Union
import math

PI = math.pi


class labelNode:
    def __init__(
        self,
        value: Union[float, str],
        means: list[float],
        variance: list[float],
        prob: float,
    ):
        self.value = value
        self.means = means
        self.variance = variance
        self.prob = prob


class NaiveBayes:
    def __init__(self):
        self.label_nodes: list[labelNode] = []

    def mean(self, x: np.ndarray):
        return np.mean(x)

    def variance(self, x: np.ndarray):
        return np.var(x, ddof=1)

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        classes, count = np.unique(y, return_counts=True)
        probs_y = count / n_samples
        for idx, label in enumerate(classes):
            means = []
            variances = []
            for feature in range(n_features):
                row_indices = y == label
                label_feature = x[row_indices, feature]
                means.append(self.mean(label_feature))
                variances.append(self.variance(label_feature) + 1e-9)
            self.label_nodes.append(labelNode(label, means, variances, probs_y[idx]))

    def calculate_probs(self, node: labelNode, x: np.ndarray):
        variances = np.array(node.variance)
        means = np.array(node.means)
        probs = np.log(node.prob)
        z1 = (-1 / 2) * np.sum(np.log(2 * PI * variances))
        z2 = (-1 / 2) * np.sum(((x - means) ** 2) / variances)
        return probs + z1 + z2

    def _predict_one(self, x: np.ndarray):
        probs = [self.calculate_probs(node, x) for node in self.label_nodes]
        return self.label_nodes[np.argmax(probs)].value

    def predict(self, x: np.ndarray):
        return np.array([self._predict_one(sample) for sample in x])


# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x, y = make_classification(
    n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
# %%

nb = NaiveBayes()
nb.fit(x_train, y_train)
y_preds = nb.predict(x_test)

print(y_preds)

print(f"accuracy : {accuracy_score(y_test,y_preds)}")
# %%
from sklearn.naive_bayes import GaussianNB

nbs = GaussianNB()
nbs.fit(x_train, y_train)
y_preds = nbs.predict(x_test)

y_preds.shape
print(f"accuracy of sklearn model: {accuracy_score(y_test,y_preds)}")
