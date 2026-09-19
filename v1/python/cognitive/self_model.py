import numpy as np
from collections import deque
from python.utils.tensor_utils import to_fixed_dim, normalize, cosine_sim, blend_vectors


class SelfModel:
    def __init__(self, population, dim=256):
        self.pop = population
        self.dim = dim

        self.self_vector = np.zeros(dim, dtype=np.float32)
        self.predicted_self = np.zeros(dim, dtype=np.float32)
        self.prediction_error = 0.0
        self.prediction_error_history = deque(maxlen=100)
        self.error_derivative = 0.0

        self.arousal = 0.3
        self.valence = 0.5
        self.certainty = 0.5
        self.energy = 0.5
        self.curiosity = 0.5
        self.focus = 0.5

        self.knowledge_base = {}
        self.belief_confidence = {}
        self.body_schema = {
            'neuron_count': population.size(),
            'core_range': (0, int(population.size() * 0.3)),
            'input_range': (int(population.size() * 0.3), int(population.size() * 0.45)),
            'output_range': (int(population.size() * 0.85), population.size()),
        }

        self.self_boundary_strength = 0.5
        self.self_generated_patterns = deque(maxlen=50)
        self.external_patterns = deque(maxlen=50)
        self._boundary_confidence = 0.5

        self._predictor_weights = np.random.normal(0, 0.05, (dim, dim)).astype(np.float32)
        self._predictor_bias = np.zeros(dim, dtype=np.float32)
        self._last_gw_vector = np.zeros(dim, dtype=np.float32)

    def update(self, step, dopamine, firing_rate, prediction_error,
               input_text=None):
        self.arousal = 0.9 * self.arousal + 0.1 * (dopamine + firing_rate)
        self.valence = 0.85 * self.valence + 0.15 * dopamine
        self.certainty = 0.9 * self.certainty + 0.1 * (1.0 - prediction_error)
        self.certainty = max(0.05, min(1.0, self.certainty))
        self.energy = 0.95 * self.energy + 0.05 * firing_rate
        self.curiosity = 0.9 * self.curiosity + 0.1 * prediction_error
        self.focus = 0.85 * self.focus + 0.15 * (1.0 - prediction_error)

        components = [
            self._encode_arousal(),
            self._encode_valence(),
            self._encode_certainty(),
            self._encode_energy(),
            self._encode_curiosity(),
            self._encode_focus(),
        ]
        self.self_vector = normalize(sum(components))

        self._compute_prediction_error()

    def predict_self(self, gw_vector=None):
        predicted = self._predictor_weights @ self.self_vector + self._predictor_bias
        self.predicted_self = normalize(predicted)

        if gw_vector is not None:
            gw_vec = to_fixed_dim(gw_vector, self.dim)
            self.predicted_self = blend_vectors(self.predicted_self, gw_vec, 0.3)
            self._last_gw_vector = gw_vec.copy()

        return self.predicted_self.copy()

    def is_self_generated(self, neuron_ids, source_type='unknown'):
        pattern = np.zeros(self.dim, dtype=np.float32)
        for nid in neuron_ids[:20]:
            idx = nid % self.dim
            pattern[idx] += 1.0

        if source_type == 'internal':
            self.self_generated_patterns.append(pattern.copy())
            return True

        if source_type == 'external':
            self.external_patterns.append(pattern.copy())
            return False

        if len(self.self_generated_patterns) < 5 or len(self.external_patterns) < 5:
            return False

        self_sim = self._max_pattern_sim(pattern, self.self_generated_patterns)
        ext_sim = self._max_pattern_sim(pattern, self.external_patterns)

        is_self = self_sim > ext_sim and self_sim > 0.3
        if is_self:
            self.self_boundary_strength = min(1.0, self.self_boundary_strength + 0.02)
        else:
            self.self_boundary_strength = max(0.1, self.self_boundary_strength - 0.02)

        return is_self

    def am_i(self, query):
        desc = self.get_self_description()
        keywords = ['sna', 'network', 'neuron', 'spike', 'self',
                    'model', 'brain', '我的', '我', '是', '自己']
        for kw in keywords:
            if kw.lower() in str(query).lower():
                return True, self.self_boundary_strength
        return False, 0.0

    def get_self_description(self):
        vec_dim = 6
        return {
            'arousal': round(self.arousal, 3),
            'valence': round(self.valence, 3),
            'certainty': round(self.certainty, 3),
            'energy': round(self.energy, 3),
            'curiosity': round(self.curiosity, 3),
            'focus': round(self.focus, 3),
            'boundary': round(self.self_boundary_strength, 3),
            'error_derivative': round(self.error_derivative, 4),
        }

    def should_sleep(self):
        return (self.error_derivative > 0.02 and self.energy < 0.3) or self.certainty < 0.2

    def _encode_arousal(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[:self.dim//6] = self.arousal
        return vec

    def _encode_valence(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[self.dim//6:self.dim//3] = self.valence
        return vec

    def _encode_certainty(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[self.dim//3:self.dim//2] = self.certainty
        return vec

    def _encode_energy(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[self.dim//2:2*self.dim//3] = self.energy
        return vec

    def _encode_curiosity(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[2*self.dim//3:5*self.dim//6] = self.curiosity
        return vec

    def _encode_focus(self):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[5*self.dim//6:] = self.focus
        return vec

    def _compute_prediction_error(self):
        self.prediction_error = float(np.linalg.norm(
            self.self_vector - self.predicted_self))
        self.prediction_error_history.append(self.prediction_error)

        if len(self.prediction_error_history) >= 10:
            recent = list(self.prediction_error_history)
            self.error_derivative = (recent[-1] - recent[-10]) / 10.0
        else:
            self.error_derivative = 0.0

        self._update_predictor()

    def _update_predictor(self):
        lr = 0.001
        error_vec = self.self_vector - self.predicted_self
        self._predictor_weights += lr * np.outer(error_vec, self.self_vector)
        self._predictor_bias += lr * error_vec

    def _max_pattern_sim(self, pattern, pattern_set):
        if not pattern_set:
            return 0.0
        best = 0.0
        pat_norm = np.linalg.norm(pattern)
        if pat_norm < 1e-10:
            return 0.0
        for p in pattern_set:
            sim = cosine_sim(pattern, p)
            if sim > best:
                best = sim
        return best