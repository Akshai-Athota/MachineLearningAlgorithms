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


class Logistic_Regression:
    def __init__(self, features: int, epochs: int, lr: float, p: float = 0.5):
        self.w = np.random.normal(0, 1, (features, 1))
        self.b: int = 0
        self.epochs = epochs
        self.lr = lr
        self.p = p

    def probs(self, x: np.array):
        z = np.dot(x, self.w) + self.b
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp((-1) * z))

    def predict(self, x: np.array):
        y_probs = self.probs(x)
        y_preds = (y_probs >= self.p).astype(int)
        return y_preds

    def update_weights(self, x: np.array, y: np.array, y_prob: np.array):
        m = y.shape[0]
        self.w = self.w - (self.lr * np.dot(x.T, (y_prob - y))) / m
        self.b = self.b - self.lr * np.mean(y_prob - y)

    def accuracy(self, y_preds: np.array, y: np.array):
        t = y_preds == y
        return np.mean(t) * 100

    def binary_categorical_loss(self, y: np.array, y_prob: np.array):
        return -1 * np.sum(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))

    def train(self, x: np.array, y: np.array):
        accuracy = []
        loss = []
        for i in range(self.epochs):
            print(f"{i}th epoch has started")
            y_prob = self.probs(x)
            y_preds = self.predict(x)
            self.update_weights(x, y, y_prob)
            accuracy.append(self.accuracy(y_preds, y))
            loss.append(self.binary_categorical_loss(y, y_prob))
            print(f"{i}th epoch has ended")

        return self.w, self.b, loss, accuracy


# %%
logistic = Logistic_Regression(3, 700, 0.01, p=0.55)
w, b, loss, accuracy = logistic.train(x_train, y_train)

print(f"accuracy after training :{accuracy[-1]}")
# %%
predictions = logistic.predict(se.transform(x_test))
print(f"test accuracy : { logistic.accuracy(predictions,y_test)}")

# %%
from sklearn.linear_model import LogisticRegression

skelarn_model = LogisticRegression().fit(x_train, y_train)
y_preds = skelarn_model.predict(se.transform(x_test))

print(f"test accuracy by sklearn is {logistic.accuracy(y_preds.reshape(-1,1),y_test)}")

# %%
