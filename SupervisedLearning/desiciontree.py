# %%
import numpy as np
from typing import Union, Optional


def probabilities(x: np.ndarray) -> np.ndarray:
    classses, count = np.unique(x, return_counts=True)
    return count / count.sum()


def gini_entropy(x: np.ndarray) -> float:
    return 1 - np.sum(probabilities(x) ** 2)


def entropy(x: np.ndarray):
    p = probabilities(x)
    return -np.sum(p * np.log2(p + 1e-9))


class Node:
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[Union[int, float]] = None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.root = None


class Decision_Tree_Classifier:

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "entropy",
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = gini_entropy if criterion == "gini" else entropy

    def _leaf_value(self, x: np.ndarray):
        classes, count = np.unique(x, return_counts=True)
        return classes[np.argmax(count)]

    def _best_fit(self, x: np.ndarray, y: np.ndarray):
        best_gain = -1
        feature_Index = None
        threshold = None
        n_samples, n_features = x.shape
        parent_impurity = self.criterion(y)

        for feature_index in range(n_features):
            thresholds = np.unique(x[:, feature_index])

            for thres in thresholds:
                left_mask = x[:, feature_index] <= thres
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_split
                    or right_mask.sum() < self.min_samples_split
                ):
                    continue

                left_criterion = self.criterion(y[left_mask])
                right_criterion = self.criterion(y[right_mask])
                weighted_criterion = (
                    (left_mask).sum() * left_criterion
                    + (right_mask.sum()) * right_criterion
                ) / n_samples
                gain = parent_impurity - weighted_criterion

                if best_gain < gain:
                    best_gain = gain
                    threshold = thres
                    feature_Index = feature_index

        return feature_Index, threshold

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0):
        n_samples, n_features = x.shape
        n_labels = np.unique(y).sum()

        if (
            n_samples < self.min_samples_split
            or n_features == 1
            or n_labels == 1
            or depth >= self.max_depth
        ):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        feature_index, threshold = self._best_fit(x, y)

        if feature_index is None:
            return Node(value=self._leaf_value(y))

        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(x[left_mask], y[left_mask], depth=depth + 1)
        right_node = self._build_tree(x[right_mask], y[right_mask], depth=depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(x, y, depth=0)

    def _predict_one(self, x, node: Node):
        if node.value is not None:
            return node.value
        elif x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, x: np.ndarray):
        y = [self._predict_one(data, self.root) for data in x]
        return np.array(y)


# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
# %%

x_train[:5]
# %%
y_train[:5]
# %%

decisiontree = Decision_Tree_Classifier()
decisiontree.fit(x_train, y_train)
# %%
y_preds = decisiontree.predict(x_test)
# %%


def accuracy(a: np.ndarray, b: np.ndarray) -> float:
    return (((a == b).sum()) / a.shape[0]) * 100


print(f"accuracy : {accuracy(y_preds,y_test)}")

# %%
from sklearn.tree import DecisionTreeClassifier

dt_sl = DecisionTreeClassifier(criterion="entropy")
dt_sl.fit(x_train, y_train)
z_preds = dt_sl.predict(x_test)
print(f"accuracy of sklearn model {accuracy(z_preds,y_test)}")

# %%
