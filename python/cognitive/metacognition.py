import random
import math


class MetaCognition:
    def __init__(self, population):
        self.pop = population
        self.uncertainty_threshold = 0.6
        self.knowledge_gaps = set()
        self.self_question_interval = 500

        self.uncertainty_history = []
        self.last_question_step = 0
        self.questions_generated = 0

    def should_ask(self, step):
        if step - self.last_question_step < self.self_question_interval:
            return False

        uncertainty = self._measure_uncertainty()

        self.uncertainty_history.append(uncertainty)
        if len(self.uncertainty_history) > 100:
            self.uncertainty_history.pop(0)

        if uncertainty > self.uncertainty_threshold:
            self.last_question_step = step
            return True

        if step % self.self_question_interval == 0:
            self.last_question_step = step
            return True

        return False

    def _measure_uncertainty(self):
        fires = self.pop.get_current_fires()
        if len(fires) < 3:
            return random.uniform(0.4, 0.9)

        strengths = [f.strength for f in fires]
        mean_strength = sum(strengths) / len(strengths)
        variance = sum((s - mean_strength) ** 2 for s in strengths) / len(strengths)

        normalized_variance = min(1.0, variance / 0.5)

        activity_level = len(fires) / max(1, self.pop.size())
        activity_uncertainty = 1.0 - min(1.0, activity_level * 10)

        return 0.6 * normalized_variance + 0.4 * activity_uncertainty

    def generate_request(self):
        self.questions_generated += 1

        request_spikes = []

        fires = self.pop.get_current_fires()
        if fires:
            main_neuron = max(fires, key=lambda f: f.strength).neuron_id
            request_neuron = main_neuron ^ 0x5555
            request_spikes.append({
                'target_id': request_neuron % self.pop.size(),
                'strength': 0.5,
                'time_ms': 0,
                'group_hint': (request_neuron * self.questions_generated) % 256
            })

        for _ in range(3):
            nid = random.randint(0, int(self.pop.size() * 0.3) - 1)
            request_spikes.append({
                'target_id': nid,
                'strength': 0.3,
                'time_ms': 1,
                'group_hint': (self.questions_generated * 31) % 256
            })

        return request_spikes

    def evaluate_command_success(self, command_output):
        if command_output and len(command_output) > 0:
            has_error = 'error' in str(command_output).lower()
            return 0.8 if not has_error else -0.2
        return -0.1
