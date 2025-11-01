from agd import agd_history
from sgd import sgd_history
from momentumgd import mgd_history
import matplotlib.pyplot as plt
import numpy as np

sgd_history = np.array(sgd_history)
agd_history = np.array(agd_history)
mgd_history = np.array(mgd_history)

plt.figure(figsize=(10, 5))
plt.plot(sgd_history[:, 1], label="SGD", marker="o")
plt.plot(mgd_history[:, 1], label="Momentum", marker="x")
plt.plot(agd_history[:, 1], label="Adam", marker="s")
plt.xlabel("Update Step")
plt.ylabel("Theta1 value")
plt.title("Parameter Convergence")
plt.legend()
plt.grid(True)
plt.show()
