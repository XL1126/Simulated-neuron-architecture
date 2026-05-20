import math
from collections import defaultdict


class Attention:
    def __init__(self, population, num_attention_neurons=100):
        self.pop = population
        self.num_attention_neurons = num_attention_neurons
        self.attention_gains = defaultdict(float)
        self.history_frequency = defaultdict(int)
        self.total_events = 0
        self._reserve_attention_neurons()

    def _reserve_attention_neurons(self):
        pass

    def compute_gains(self):
        fires = self.pop.get_current_fires()
        for f in fires:
            self.history_frequency[f.neuron_id] += 1
        self.total_events += len(fires)

        if self.total_events > 1000000:
            for k in self.history_frequency:
                self.history_frequency[k] //= 2
            self.total_events //= 2

        for f in fires:
            frequency = self.history_frequency.get(f.neuron_id, 0)
            expected = max(1, self.total_events / (self.pop.size() or 1))

            novelty = 1.0 - min(1.0, frequency / (expected * 5))

            recency_bonus = 0.3

            gain = 0.3 + 0.4 * novelty + 0.3 * recency_bonus

            self.attention_gains[f.neuron_id] = gain
            f.strength *= gain

    def get_gain(self, neuron_id):
        return self.attention_gains.get(neuron_id, 0.5)

    def reset(self):
        self.attention_gains.clear()
