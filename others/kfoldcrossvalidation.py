import numpy as np
from sklearn.metrics import accuracy_score


def k_fold(n_samples, k):
    indicies = np.arange(n_samples)
    np.random.shuffle(indicies)

    folds = []
    fold_size = n_samples // k
    for i in range(k):
        folds.append(indicies[i * (fold_size) : (i + 1) * fold_size])

    if fold_size * k <= n_samples:
        folds.append(indicies[(k - 1) * fold_size :])

    return folds


def k_fold_train(model, x, y, k, metric=accuracy_score):
    n_samples = x.shape[0]
    folds = k_fold(n_samples, k)
    n_folds = len(folds)
    scores = []

    for k in range(n_folds):
        x_val, y_val = x[folds[k]], y[folds[k]]
        train_idx = np.hstack([folds[i] for i in range(n_folds) if i != k])
        x_train, y_train = x[train_idx], y[train_idx]
        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        scores.append(metric(y_val, y_preds))

    return scores


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
lr = LogisticRegression()
all_scores = k_fold_train(
    lr,
    X,
    y,
    k=5,
)
print(all_scores)
