# %%
import numpy as np

np.random.seed(42)


class LassoRegression:
    def __init__(self, epochs: int = 100, lr: float = 0.001, cost: float = 0.1):
        self.epochs = epochs
        self.lr = lr
        self.cost = cost
        self.w = None
        self.b = 0.0

    def predict(self, x: np.ndarray):
        return np.dot(x, self.w) + self.b

    def update_weights(self, x: np.ndarray, y: np.ndarray, y_preds: np.ndarray):
        n = x.shape[0]
        dw = (-2 / n) * np.dot(x.T, (y - y_preds)) + self.cost * np.sign(self.w)
        db = -2 * np.mean(y - y_preds)
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def loss(self, y_preds: np.ndarray, y: np.ndarray):
        return np.sqrt(np.mean((y_preds - y) ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray):
        y = np.reshape(y, (-1, 1))
        n_samples, n_featrues = x.shape
        self.w = np.random.uniform(0, 1, (n_featrues, 1))
        loss = []
        for i in range(self.epochs):
            y_preds = self.predict(x)
            self.update_weights(x, y, y_preds)
            loss.append(self.loss(y_preds, y))

        return loss


# %%
import pandas as pd

# %% Load Dataset

dataset = pd.read_csv("./../Datasets/housing_linear_regression.csv")
dataset.head(5)

# %% Dataset Info

dataset.describe()
dataset.info()

# %% Check ocean_proximity

dataset["ocean_proximity"].describe()
dataset["ocean_proximity"].unique()

# %% One Hot Encode ocean_proximity

from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder(drop="first", sparse_output=False)
encoded_array = oe.fit_transform(dataset[["ocean_proximity"]])

dataset.drop(columns=["ocean_proximity"], inplace=True)
encoded_dataframe = pd.DataFrame(
    encoded_array, columns=oe.get_feature_names_out(["ocean_proximity"])
)

dataset = pd.concat(
    [dataset.reset_index(drop=True), encoded_dataframe.reset_index(drop=True)], axis=1
)
dataset.head()

# %% Features and Target

X = dataset.drop("median_house_value", axis=1)
y = dataset["median_house_value"]
columns = X.columns

# %% Train-Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# %% Impute Missing Values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=columns)
x_test = pd.DataFrame(imputer.transform(x_test), columns=columns)

# %% Add Feature Engineering Columns


def add_columns(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


x_train = add_columns(x_train)
x_test = add_columns(x_test)

# %% Drop Multicollinear Features based on VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

features = x_train[
    [
        "total_rooms",
        "rooms_per_household",
        "households",
        "population",
        "bedrooms_per_room",
        "population_per_household",
        "total_bedrooms",
    ]
]

vif = pd.DataFrame()
vif["feature"] = features.columns
vif["VIF"] = [
    variance_inflation_factor(features.values, i) for i in range(features.shape[1])
]
print(vif)

x_train.drop(columns=["total_rooms", "households", "total_bedrooms"], inplace=True)
x_test.drop(columns=["total_rooms", "households", "total_bedrooms"], inplace=True)

# %% Correlation Check

corr = x_train.join(y_train).corr()["median_house_value"]
print(corr)

# %% Standardize Features and Target

from sklearn.preprocessing import StandardScaler

x_transform = StandardScaler()
x_train_scaled = x_transform.fit_transform(x_train)
x_test_scaled = x_transform.transform(x_test)

y_train = y_train.astype(float)
y_transform = StandardScaler()
y_train_scaled = y_transform.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_transform.transform(y_test.values.reshape(-1, 1)).ravel()

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%

rr = LassoRegression(epochs=5000, lr=0.001, cost=0.02)
loss = rr.fit(x_train_scaled, y_train_scaled)

print(f"final loss of the model : {loss[-1]}")
# %%
y_preds = rr.predict(x_test_scaled)
loss = rr.loss(y_test_scaled, y_preds)
y_actual_preds = y_transform.inverse_transform(np.reshape(y_preds, (-1, 1)))
print(f"test loss : {loss}")
print(y_actual_preds)
# %%
print(y_test)
