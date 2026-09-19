import math
from collections import deque


class IntrinsicMotivation:
    def __init__(self):
        self.pattern_memory = deque(maxlen=1000)
        self.novelty_baseline = 0.0
        self.novelty_decay = 0.999
        self.info_gain_weight = 0.5
        self.consistency_weight = 0.3
        self.curiosity_weight = 0.2

    def compute(self, prediction_error, novelty=None, active_pattern=None):
        reward = 0.0

        if prediction_error is not None:
            error_reward = max(0.0, 1.0 - prediction_error * 2)
            reward += error_reward * 0.4

        if active_pattern is not None:
            info_gain = self._compute_info_gain(active_pattern)
            reward += info_gain * self.info_gain_weight

            consistency = self._compute_consistency(active_pattern)
            reward += consistency * self.consistency_weight

            curiosity = self._compute_curiosity()
            reward += curiosity * self.curiosity_weight

            self.pattern_memory.append(active_pattern)

        return max(0.0, min(1.0, reward))

    def _compute_info_gain(self, pattern):
        if len(self.pattern_memory) < 10:
            return 0.5

        pattern_set = self._pattern_to_set(pattern)

        similarities = []
        for mem_pattern in list(self.pattern_memory)[-50:]:
            mem_set = self._pattern_to_set(mem_pattern)
            if not mem_set:
                continue
            intersect = len(pattern_set & mem_set)
            union = len(pattern_set | mem_set)
            sim = intersect / union if union > 0 else 0.0
            similarities.append(sim)

        if not similarities:
            return 0.5

        avg_sim = sum(similarities) / len(similarities)
        return 1.0 - avg_sim

    def _pattern_to_set(self, pattern):
        if not pattern:
            return set()
        if isinstance(pattern, list):
            if len(pattern) > 0:
                if hasattr(pattern[0], 'neuron_id'):
                    return {p.neuron_id for p in pattern}
                elif isinstance(pattern[0], (int, float)):
                    return {int(p) for p in pattern}
        return set()

    def _compute_consistency(self, pattern):
        if len(self.pattern_memory) < 20:
            return 0.0

        pattern_set = self._pattern_to_set(pattern)

        recent = list(self.pattern_memory)[-10:]
        consistent_count = 0

        for mem in recent:
            mem_set = self._pattern_to_set(mem)
            if mem_set and pattern_set & mem_set:
                consistent_count += 1

        return consistent_count / max(1, len(recent))

    def _compute_curiosity(self):
        self.novelty_baseline = self.novelty_baseline * self.novelty_decay
        if len(self.pattern_memory) < 5:
            return 0.5

        exploration_value = 1.0 / (1.0 + math.log(len(self.pattern_memory) + 1))
        return exploration_value * 0.5
