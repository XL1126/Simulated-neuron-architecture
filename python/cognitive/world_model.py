from collections import deque, defaultdict
import math


class WorldModel:
    def __init__(self, population, hidden_size=500, prediction_window_ms=10):
        self.pop = population
        self.hidden_size = hidden_size
        self.prediction_window_ms = prediction_window_ms

        self.state_history = deque(maxlen=prediction_window_ms)
        self.prediction_weights = {}
        self.learning_rate = 0.01
        self.prediction_error = 0.0
        self.total_prediction_error = 0.0

        self._primed = False

    def predict_and_compare(self):
        current_fires = self.pop.get_current_fires()
        current_state = self._encode_state(current_fires)

        if not self._primed:
            self.state_history.append(current_state)
            if len(self.state_history) >= self.prediction_window_ms:
                self._primed = True
            return 0.0

        prediction = self._predict_next_state()

        if self.state_history:
            self.state_history.append(current_state)

        error = self._compute_prediction_error(prediction, current_state)

        self._update_weights(prediction, current_state, error)

        self.prediction_error = error
        self.total_prediction_error = self.total_prediction_error * 0.99 + error * 0.01

        return error

    def _encode_state(self, fires):
        state = {}
        for f in fires:
            state[f.neuron_id] = f.strength
        return state

    def _predict_next_state(self):
        prediction = {}
        if not self.state_history:
            return prediction

        last_state = self.state_history[-1]
        for nid, strength in last_state.items():
            pred_strength = strength * 0.8
            for hist_state in list(self.state_history)[-5:]:
                if nid in hist_state:
                    coef = self.prediction_weights.get((nid, nid), 0.0)
                    pred_strength += coef * hist_state[nid]
            prediction[nid] = pred_strength

        return prediction

    def _compute_prediction_error(self, prediction, actual):
        if not actual:
            return 0.0

        total_error = 0.0
        count = 0

        all_keys = set(prediction.keys()) | set(actual.keys())
        for nid in all_keys:
            pred_val = prediction.get(nid, 0.0)
            actual_val = actual.get(nid, 0.0)
            total_error += (pred_val - actual_val) ** 2
            count += 1

        return math.sqrt(total_error / max(1, count))

    def _update_weights(self, prediction, actual, error):
        for nid in set(prediction.keys()) | set(actual.keys()):
            pred_val = prediction.get(nid, 0.0)
            actual_val = actual.get(nid, 0.0)
            delta = actual_val - pred_val

            key = (nid, nid)
            if key not in self.prediction_weights:
                self.prediction_weights[key] = 0.0
            self.prediction_weights[key] += self.learning_rate * delta * pred_val

    def get_prediction_error(self):
        return self.total_prediction_error
