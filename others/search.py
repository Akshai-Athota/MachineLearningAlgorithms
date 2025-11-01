import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.linear_model import LogisticRegression


def cross_validation(model, x, y, k=5, metric=accuracy_score):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kfold.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        score = metric(y_val, y_preds)
        scores.append(score)

    return np.mean(scores)


def grid_search(model_class, x, y, model_params: dict, k=5, metric=accuracy_score):
    params = list(model_params.keys())
    values = list(model_params.values())

    best_score = -np.inf
    best_params = None

    for value in product(*values):
        model_param = dict(zip(params, value))
        model = model_class(**model_param)
        score = cross_validation(model, x, y, k, metric)

        print(f"Params: {model_param} -> Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = model_param

    return best_params, best_score


def random_search(model_class, param_distributions, X, y, n_iter=10, k=5):
    rng = np.random.default_rng(42)
    param_names = list(param_distributions.keys())

    best_score = -np.inf
    best_params = None

    for i in range(n_iter):
        params = {}
        for name in param_names:
            vals = param_distributions[name]
            params[name] = rng.choice(vals)
        model = model_class(**params)
        score = cross_validation(model, X, y, k=k)

        print(f"Iter {i+1}: Params: {params} -> Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


X, y = load_iris(return_X_y=True)
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]}


best_params_grid, best_score_grid = grid_search(LogisticRegression, X, y, param_grid)
print(f"\nBest Grid Search Params: {best_params_grid}, Score: {best_score_grid:.4f}")


best_params_rand, best_score_rand = random_search(
    LogisticRegression, param_grid, X, y, n_iter=8
)
print(f"\nBest Random Search Params: {best_params_rand}, Score: {best_score_rand:.4f}")
