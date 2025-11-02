import numpy as np
from hmmlearn import hmm


states = ["Rainy", "Sunny"]
n_states = len(states)


observations = ["Walk", "Shop", "Clean"]
n_observations = len(observations)


start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emit_prob = np.array(
    [
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1],
    ]
)


model = hmm.MultinomialHMM(n_components=n_states, n_iter=10, init_params="", n_trials=3)
model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emissionprob_ = emit_prob


obs_map = {"Walk": 0, "Shop": 1, "Clean": 2}
obs_sequence = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


log_prob = model.score(obs_sequence)
print(f"Log-likelihood of the observation sequence: {log_prob:.3f}")


log_prob, hidden_states = model.decode(obs_sequence, algorithm="viterbi")
print("\nMost likely hidden state sequence:")
print([states[i] for i in hidden_states])
