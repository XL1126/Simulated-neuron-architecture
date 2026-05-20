import numpy as np
from collections import deque


class PredictiveCoding:
    def __init__(self, population, num_levels=3, hidden_size=300):
        self.pop = population
        self.num_levels = num_levels
        self.hidden_size = hidden_size

        self.level_states = [np.zeros(hidden_size) for _ in range(num_levels)]
        self.level_predictions = [np.zeros(hidden_size) for _ in range(num_levels)]
        self.level_errors = [np.zeros(hidden_size) for _ in range(num_levels)]

        self.top_down_weights = [
            np.random.normal(0, 0.1 / np.sqrt(hidden_size), (hidden_size, hidden_size))
            for _ in range(num_levels - 1)
        ]
        self.bottom_up_weights = [
            np.random.normal(0, 0.1 / np.sqrt(hidden_size), (hidden_size, hidden_size))
            for _ in range(num_levels - 1)
        ]

        self.learning_rate = 0.005
        self.precision = np.ones(num_levels)
        self.state_history = deque(maxlen=50)
        self.total_free_energy = 0.0

    def encode_state(self, fires, level=0):
        vec = np.zeros(self.hidden_size)
        if not fires:
            return vec
        for f in fires[:self.hidden_size]:
            idx = f.neuron_id % self.hidden_size
            vec[idx] += f.strength
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def step(self, fires):
        current_input = self.encode_state(fires, 0)
        self.level_states[0] = 0.8 * self.level_states[0] + 0.2 * current_input

        for lvl in range(1, self.num_levels):
            prediction = np.tanh(
                self.top_down_weights[lvl - 1] @ self.level_states[lvl]
                if lvl < self.num_levels - 1
                else self.top_down_weights[lvl - 1] @ self.level_states[lvl - 1]
            )
            self.level_predictions[lvl - 1] = prediction

            error = self.level_states[lvl - 1] - prediction
            self.level_errors[lvl - 1] = error * self.precision[lvl - 1]

            self.level_states[lvl] = 0.7 * self.level_states[lvl] + 0.3 * (
                self.bottom_up_weights[lvl - 1].T @ self.level_errors[lvl - 1]
            )

        if self.num_levels > 1:
            prediction_lowest = np.tanh(
                self.top_down_weights[-1] @ self.level_states[-1])
            error = self.level_states[-2] - prediction_lowest
            self.level_errors[-2] = error * self.precision[-2]

        free_energy = 0.0
        for i in range(self.num_levels - 1):
            free_energy += np.sum(self.level_errors[i] ** 2) * self.precision[i]
            free_energy -= np.log(self.precision[i] + 1e-10)
        self.total_free_energy = self.total_free_energy * 0.99 + free_energy * 0.01

        self._update_weights()
        self.state_history.append(self.level_states[-1].copy())

        return free_energy

    def _update_weights(self):
        for lvl in range(self.num_levels - 1):
            dw_top = np.outer(self.level_errors[lvl], self.level_states[lvl + 1]
                               if lvl + 1 < self.num_levels else self.level_states[lvl])
            self.top_down_weights[lvl] += self.learning_rate * dw_top.T

            dw_bot = np.outer(self.level_states[lvl + 1]
                              if lvl + 1 < self.num_levels else self.level_states[lvl],
                              self.level_errors[lvl])
            self.bottom_up_weights[lvl] += self.learning_rate * dw_bot.T

        for lvl in range(self.num_levels):
            expected_precision = 1.0 / (np.mean(self.level_errors[lvl] ** 2) + 1.0)
            self.precision[lvl] = 0.9 * self.precision[lvl] + 0.1 * expected_precision

    def get_top_prediction(self):
        if self.num_levels > 1:
            return np.tanh(self.top_down_weights[-1] @ self.level_states[-1])
        return self.level_predictions[0]

    def get_prediction_error(self):
        return self.total_free_energy

    def get_highest_state(self):
        return self.level_states[-1]

    def compute_surprise(self, fires):
        current = self.encode_state(fires, 0)
        prediction = self.level_predictions[0] if self.level_predictions[0].any() else np.zeros(self.hidden_size)
        return np.sum((current - prediction) ** 2)

    def generate_imagined_state(self, steps=10):
        state = self.level_states[-1].copy()
        trajectory = [state.copy()]
        for _ in range(steps):
            noise = np.random.normal(0, 0.01, state.shape)
            prediction = np.tanh(self.top_down_weights[-1] @ state) if self.num_levels > 1 else state
            state = 0.9 * state + 0.1 * (prediction + noise)
            trajectory.append(state.copy())
        return trajectory