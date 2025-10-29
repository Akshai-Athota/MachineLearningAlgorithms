# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

np.random.seed(42)

X, Y = load_iris(return_X_y=True)

# %%
type(X)
# %%
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
# %%
df_x.head()
# %%
df_y.head()
# %%
df_y.describe()
# %%
classes = df_y[0].unique()
classes
# %%
df_x.describe()
# %%


def scatter_plot(x, y):
    plt.scatter(x, y)
    plt.xlabel("independent variable")
    plt.ylabel("dependent variable")
    plt.title("scatter plot")
    plt.show()


scatter_plot(df_x[0], df_y[0])
scatter_plot(df_x[1], df_y[0])
scatter_plot(df_x[2], df_y[0])
scatter_plot(df_x[3], df_y[0])
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, random_state=42)
# %%
print(f"shape of x_train {x_train.shape}")
print(f"shape of y_train {y_train.shape}")
print(f"shape of x_test {x_test.shape}")
print(f"shape of y_test {y_test.shape}")
# %%
import statistics


class KNN:
    def __init__(self, x: np.array, y: np.array, k: int):
        self.x = x
        self.y = y
        self.k = k

    def distance(self, x, y):
        return np.sum((x - y) ** 2)

    def predict(self, x: np.array):
        y_preds = list()
        for z in x:
            distances = list()
            for i, point in enumerate(self.x):
                distances.append([self.distance(point, z), self.y[i]])
            distances = sorted(distances, reverse=False)
            neighbours = [i[1] for i in distances[: self.k]]
            y_pred = statistics.mode(neighbours)
            y_preds.append(y_pred)

        return np.array(y_preds)

    def accuracy(self, y: np.array, y_preds: np.array):
        accurate = y == y_preds
        return ((accurate.sum()) / y.shape[0]) * 100


knn_5 = KNN(x_train, y_train, 5)

y_preds = knn_5.predict(x_test)

print(f"predictions by model_5 {y_preds}")
print(f"actual values of y {y_test}")

print(f"accuracy with neighbours 5 :{knn_5.accuracy(y_test,y_preds)}")

# %%
knn_3 = KNN(x_train, y_train, 3)

y_preds = knn_3.predict(x_test)

print(f"predictions by model_3 {y_preds}")

print(f"accuracy with neighbours 3 :{knn_3.accuracy(y_test,y_preds)}")


# %%
knn_2 = KNN(x_train, y_train, 2)

y_preds = knn_2.predict(x_test)

print(f"predictions by model_2 {y_preds}")

print(f"accuracy with neighbours 2 :{knn_2.accuracy(y_test,y_preds)}")
# %%
knn_1 = KNN(x_train, y_train, 1)

y_preds = knn_1.predict(x_test)

print(f"predictions by model_1 {y_preds}")

print(f"accuracy with neighbours 1 :{knn_1.accuracy(y_test,y_preds)}")
# %%


from sklearn.neighbors import KNeighborsClassifier

sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(x_train, y_train)
y_preds = sklearn_knn.predict(x_test)
print(f"accuracy of the sklearn model {knn_5.accuracy(y_preds,y_test)}")
# %%
