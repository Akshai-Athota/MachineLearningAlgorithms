import numpy as np
from loss_grad import grad_i, theta as ta

np.random.seed(42)


class AGD:

    def __init__(self, epochs=100, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def agd(self, x, y):
        theta = np.array(ta, dtype=float)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        history = [theta.copy()]

        N = len(x)
        t = 0

        for _ in range(self.epochs):
            for i in range(N):
                t += 1
                g = grad_i(theta, x[i], y[i])

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g**2)

                m_hat = m / (1 - self.beta1**t)
                v_hat = v / (1 - self.beta2**t)

                theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                history.append(theta.copy())

        return theta, history


x = np.arange(10, dtype=int)
y = x * 2 + 1


agd = AGD()
tt, agd_history = agd.agd(x, y)

print(agd_history)
