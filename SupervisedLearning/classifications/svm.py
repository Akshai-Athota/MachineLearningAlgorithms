# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cur_direct = os.path.dirname(__file__)
dataset_path = os.path.join(cur_direct, "..", "Datasets", "Social_Network_Ads.csv")
df = pd.read_csv(dataset_path)

df.head()
# %%
df.describe()
# %%
columns = df.columns
columns
# %%
df.drop(columns=["User ID"], inplace=True)

df.head()
# %%
np.random.seed(42)

from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder(drop="first", sparse_output=False)
encode_data = oe.fit_transform(df[["Gender"]])
encode_data = pd.DataFrame(encode_data, columns=oe.get_feature_names_out(["Gender"]))
# %%
df.drop(columns=["Gender"], inplace=True)

encoded_df = pd.concat([df, encode_data], axis=1)

encoded_df.head()
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    encoded_df[["Age", "Gender_Male", "EstimatedSalary"]],
    encoded_df[["Purchased"]],
    shuffle=True,
)

print(f"x train shape : {x_train.shape}")
print(f"x test shape : {x_test.shape}")
print(f"y train shape : {y_train.shape}")
print(f"y train shape : {y_test.shape}")
# %%

from sklearn.preprocessing import StandardScaler

se = StandardScaler()
x_train = se.fit_transform(x_train)

x_train_df = pd.DataFrame(x_train, columns=["Age", "Gender_Male", "EstimatedSalary"])
x_train_df.describe()


# %%
correlations = x_train_df.corr()
correlations

# %%
np.random.seed(42)


class SVMLinear:

    def __init__(self, epochs: int, lr: float, cost: float):
        self.w = None
        self.b = 0.0
        self.epcohs = epochs
        self.lr = lr
        self.cost = cost

    def fit(self, x: np.array, y: np.array):
        n, m = x.shape
        y = y.reshape(-1, 1)
        self.w = np.random.normal(0, 1, (m, 1))
        for idx, x_i in enumerate(x):
            x_i = x_i.reshape(-1, 1)
            constarin = (y[idx] * (np.dot(self.w.T, x_i) + self.b)) >= 1
            if constarin:
                self.w -= self.lr * (self.w)
            else:
                self.w -= self.lr * (
                    2 * self.cost * self.w - (np.dot(x_i, y[idx])).reshape(-1, 1)
                )
                self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        y_preds = np.sign(approx)
        y_preds[y_preds == -1] = 0
        return y_preds

    def accuracy(self, y: np.array, y_pred: np.array):
        z = (y == y_pred).mean() * 100
        return z.astype(int)


# %%

svm = SVMLinear(100, 0.001, 0.2)
svm.fit(x_train, y_train.to_numpy())
y_preds = svm.predict(se.transform(x_test))
print(y_preds)
# %%
print(f" {svm.accuracy(y_test,y_preds)}")

# %%

from sklearn.svm import SVC

svm_sklearn = SVC(kernel="linear", C=0.2)

svm_sklearn.fit(x_train, y_train.to_numpy())

y_preds = svm_sklearn.predict(se.transform(x_test))

print(f"accuracy of the sklearn model : {svm.accuracy(y_test,y_preds.reshape(-1,1))}")
# %%
