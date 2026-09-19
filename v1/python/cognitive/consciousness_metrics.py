import numpy as np
from collections import deque, defaultdict

from .unified_consciousness import compute_consciousness_level, cross_validate_consciousness, compute_consciousness_diagnostic_report


class ConsciousnessMetrics:
    def __init__(self, gw_population_size=200):
        self.gw_size = gw_population_size

        self.phi = 0.0
        self.phi_history = deque(maxlen=100)

        self.gw_participation = {}
        self.gw_participation_history = defaultdict(deque)

        self.self_prediction_error = 0.0
        self.self_error_history = deque(maxlen=100)
        self.self_error_derivative = 0.0

        self.consciousness_level = 0.0
        self.level_history = deque(maxlen=100)

        self.first_person_salience = 0.0
        self.perceptual_vividness = 0.0

        self.module_labels = ['sensory', 'semantic', 'memory',
                              'action', 'self_reflection']

    def update_phi(self, phi_value):
        self.phi = float(phi_value)
        self.phi_history.append(self.phi)

    def update_gw_participation(self, module_name, duration_ms):
        if module_name not in self.gw_participation:
            self.gw_participation[module_name] = 0.0
            self.gw_participation_history[module_name] = deque(maxlen=50)

        self.gw_participation[module_name] += duration_ms
        self.gw_participation_history[module_name].append(duration_ms)

        total = sum(self.gw_participation.values()) + 1.0
        for k in self.gw_participation:
            self.gw_participation[k] *= 0.99

    def update_self_error(self, error_value):
        self.self_prediction_error = float(error_value)
        self.self_error_history.append(self.self_prediction_error)

        if len(self.self_error_history) >= 10:
            recent = list(self.self_error_history)
            self.self_error_derivative = (recent[-1] - recent[-10]) / 10.0

    def update_qualia(self, first_person_salience=0.0, perceptual_vividness=0.0):
        self.first_person_salience = float(first_person_salience)
        self.perceptual_vividness = float(perceptual_vividness)

    def compute_consciousness_level(self, gw_activity, self_model=None):
        self.consciousness_level = compute_consciousness_level(
            phi=self.phi,
            global_ignition=gw_activity,
            self_prediction_error=self.self_prediction_error,
            first_person_salience=self.first_person_salience,
            perceptual_vividness=self.perceptual_vividness,
        )
        self.level_history.append(self.consciousness_level)

        return self.consciousness_level

    def partition_bipartition(self, activity_vector):
        n = len(activity_vector)
        if n < 2:
            return 0.0

        mid = n // 2
        part_a = activity_vector[:mid]
        part_b = activity_vector[mid:]

        h_a = -np.sum(part_a * np.log(part_a + 1e-10))
        h_b = -np.sum(part_b * np.log(part_b + 1e-10))
        h_ab = -np.sum(activity_vector * np.log(activity_vector + 1e-10))

        mi = h_a + h_b - h_ab
        phi_approx = np.exp(-mi) if mi > 0 else 0.0
        return float(phi_approx)

    def get_consciousness_report(self):
        return {
            'phi': round(self.phi, 4),
            'phi_trend': self._compute_trend(self.phi_history),
            'consciousness_level': round(self.consciousness_level, 3),
            'self_prediction_error': round(self.self_prediction_error, 4),
            'self_error_derivative': round(self.self_error_derivative, 4),
            'gw_participation': {k: round(v, 3) for k, v
                in sorted(self.gw_participation.items(),
                         key=lambda x: x[1], reverse=True)},
            'level_trend': self._compute_trend(self.level_history),
        }

    def _compute_trend(self, history):
        if len(history) < 20:
            return 'stable'
        recent = list(history)
        first_half = np.mean(list(recent)[:len(recent)//2])
        second_half = np.mean(list(recent)[len(recent)//2:])
        diff = second_half - first_half
        if diff > 0.01:
            return 'rising'
        elif diff < -0.01:
            return 'falling'
        return 'stable'

    def get_metrics_for_dashboard(self, gw_winner, ignition, winner_concept=''):
        return {
            'phi': round(self.phi, 4),
            'consciousness': round(self.consciousness_level, 3),
            'gw_winner': gw_winner,
            'gw_winner_concept': winner_concept,
            'gw_ignited': ignition,
            'self_error': round(self.self_prediction_error, 4),
            'self_error_d': round(self.self_error_derivative, 4),
        }