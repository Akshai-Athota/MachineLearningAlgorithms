import numpy as np

states = ["Sunny", "Rainy", "Cloudy"]

P = np.array([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.2, 0.4, 0.4]])


pi = np.array([0.5, 0.3, 0.2])


def next_state(curr_state):
    return np.random.choice(states, p=P[states.index(curr_state)])


np.random.seed(42)
days = ["Day " + str(i + 1) for i in range(10)]
curr_state = np.random.choice(states, p=pi)
sequence = [curr_state]

for _ in range(9):
    curr_state = np.random.choice(states, p=P[states.index(curr_state)])
    sequence.append(curr_state)

for d, s in zip(days, sequence):
    print(f"{d}: {s}")

from numpy.linalg import matrix_power

pi_t = pi @ matrix_power(P, 50)  # after 50 steps
print("Steady-state distribution:", pi_t)
