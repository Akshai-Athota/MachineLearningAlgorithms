import numpy as np
from loss_grad import grad_i, theta as t

np.random.seed(42)


class Momentum:

    def __init__(self, epochs=100, lr=0.01, beta=0.9):
        self.epochs = epochs
        self.lr = lr
        self.beta = beta

    def momentum(self, x, y):
        theta = np.array(t, dtype=float)
        v = np.zeros_like(theta)
        history = [theta.copy()]

        N = len(x)

        for _ in range(self.epochs):
            for i in range(N):
                g = grad_i(theta, x[i], y[i])
                v = self.beta * v + (1 - self.beta) * g
                theta -= self.lr * v
                history.append(theta.copy())

        return theta, history


x = np.arange(10, dtype=int)
y = x * 2 + 1


mgd = Momentum()
tt, mgd_history = mgd.momentum(x, y)

print(mgd_history)
