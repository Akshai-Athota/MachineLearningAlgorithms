# %%
import numpy as np
from SupervisedLearning.classifications.desiciontree import (
    Decision_Tree_Classifier,
    accuracy,
)
from typing import Optional, Union
from collections import Counter


class Random_Forest:
    def __init__(
        self,
        max_trees: Optional[int] = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "entropy",
        max_features: Optional[int] = None,
    ):
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []

    def _bootstrap_samples(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return x[indices], y[indices]

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.trees = []

        for nth_tree in range(self.max_trees):
            print(f"{nth_tree} th tree is training")
            x_samples, y_samples = self._bootstrap_samples(x, y)
            tree = Decision_Tree_Classifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )

            if self.max_features is None:
                self.max_features = int(np.sqrt(x_samples.shape[1]))

            feature_indices = np.random.choice(
                x_samples.shape[1], self.max_features, replace=False
            )

            tree.fit(x_samples[:, feature_indices], y_samples)
            self.trees.append((tree, feature_indices))

    def predict(self, x: np.ndarray):
        tree_preds = np.array(
            [
                tree.predict(x[:, feature_indcies])
                for tree, feature_indcies in self.trees
            ]
        )

        y_preds = np.array(
            [Counter(preds).most_common(1)[0][0] for preds in tree_preds.T]
        )

        return y_preds


# %%

from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

rfc = Random_Forest()
rfc.fit(x_train, y_train)
y_preds = rfc.predict(x_test)

y_preds.shape, y_test.shape

# %%
print(f"accuracy of own model:{accuracy(y_preds,y_test)}")
# %%


from sklearn.ensemble import RandomForestClassifier

rfc_sklearn = RandomForestClassifier(n_estimators=5, criterion="entropy")
rfc_sklearn.fit(x_train, y_train)
y_preds = rfc.predict(x_test)

print(f"accuracy of sklearn model:{accuracy(y_preds,y_test)}")

# %%
