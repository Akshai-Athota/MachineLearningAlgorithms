import numpy as np


def predict(theta, x):
    return theta[0] + theta[1] * x


def grad_i(theta, x_i, y_i):
    y_hat = predict(theta, x_i)
    dtheta0 = 2 * (y_hat - y_i)
    dtheta1 = 2 * (y_hat - y_i) * x_i
    return np.array([dtheta0, dtheta1])


theta = [0, 0]
