import numpy as np
from loss_grad import grad_i, theta as t

np.random.seed(42)


class SGD:

    def __init__(self, epochs=100, lr=0.01):
        self.epochs = epochs
        self.lr = lr

    def sgd(self, x, y):
        theta = np.array(t, dtype=float)
        history = [theta.copy()]

        N = len(x)

        for _ in range(self.epochs):
            for i in range(N):
                g = grad_i(theta, x[i], y[i])
                theta -= self.lr * g
                history.append(theta.copy())

        return theta, history


x = np.arange(10, dtype=int)
y = x * 2 + 1


sgd = SGD()
tt, sgd_history = sgd.sgd(x, y)

print(sgd_history)
