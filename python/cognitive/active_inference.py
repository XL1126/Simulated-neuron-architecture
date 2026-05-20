import numpy as np
from collections import deque


class ActiveInference:
    def __init__(self, population, predictive_coding, spa, dim=256):
        self.pop = population
        self.pc = predictive_coding
        self.spa = spa
        self.dim = dim

        self.action_repertoire = {
            'wait': lambda: np.zeros(dim),
            'explore_virtual': lambda: self._random_action_vector(),
            'internal_query': lambda: self.pc.generate_imagined_state(3)[-1],
            'recall_memory': lambda: self._recall_vector(),
            'generate_output': lambda: self._output_vector(),
            'self_reflect': lambda: self._self_reflect_vector(),
        }

        self.action_values = {k: 0.5 for k in self.action_repertoire}
        self.action_history = deque(maxlen=50)
        self.expected_free_energy = {k: 0.5 for k in self.action_repertoire}
        self.policy_precision = 1.0
        self.chosen_action = None
        self.novelty_seeking = 0.3
        self.total_actions = 0

    def select_action(self, workspace_state=None):
        if workspace_state is not None:
            for action_name in self.action_repertoire:
                action_vec = self.action_repertoire[action_name]()
                if len(action_vec) > self.dim:
                    action_vec = action_vec[:self.dim]
                elif len(action_vec) < self.dim:
                    action_vec = np.pad(action_vec, (0, self.dim - len(action_vec)))

                ws_vec = workspace_state[:min(len(workspace_state), self.dim)]
                ws_vec = np.pad(ws_vec, (0, max(0, self.dim - len(ws_vec))))
                ws_vec = ws_vec[:self.dim]

                alignment = np.dot(action_vec, ws_vec) / (
                    np.linalg.norm(action_vec) * np.linalg.norm(ws_vec) + 1e-10)
                self.expected_free_energy[action_name] = (
                    self.expected_free_energy[action_name] * 0.9 +
                    (1.0 - abs(alignment)) * 0.1)

        probabilities = []
        actions = []
        for name, value in self.action_values.items():
            efg = self.expected_free_energy.get(name, 0.5)
            prob = np.exp(self.policy_precision * (value - efg))
            probabilities.append(prob)
            actions.append(name)

        probs = np.array(probabilities)
        probs = probs / probs.sum()

        chosen = np.random.choice(actions, p=probs)
        self.chosen_action = chosen
        self.total_actions += 1
        self.action_history.append(chosen)

        return chosen, self.action_repertoire[chosen]()

    def update_value(self, action_name, reward, prediction_error):
        td_error = reward - self.action_values[action_name]
        self.action_values[action_name] += 0.05 * td_error

        self.expected_free_energy[action_name] = (
            self.expected_free_energy[action_name] * 0.95 +
            prediction_error * 0.05)

        if reward > 0.5:
            self.policy_precision = min(3.0, self.policy_precision + 0.05)
        else:
            self.policy_precision = max(0.5, self.policy_precision - 0.02)

    def _random_action_vector(self):
        return np.random.normal(0, 0.5, self.dim)

    def _recall_vector(self):
        return self.pc.get_highest_state()

    def _output_vector(self):
        return self.pc.get_top_prediction()

    def _self_reflect_vector(self):
        thought = self.pc.get_highest_state()
        return thought * np.random.uniform(0.8, 1.2, len(thought))

    def get_policy_summary(self):
        return {
            'chosen_action': self.chosen_action,
            'action_values': dict(self.action_values),
            'policy_precision': round(self.policy_precision, 3),
            'total_actions': self.total_actions,
        }

    def inject_action_spikes(self, action_vector, strength=0.5):
        events = []
        indices = np.argsort(np.abs(action_vector))[::-1]
        for i in indices[:10]:
            nid = int(abs(action_vector[i]) * 10000) % self.pop.size()
            events.append({
                'target_id': nid,
                'strength': strength * abs(action_vector[i]),
                'time_ms': 0,
                'group_hint': i % 256,
            })
        return events