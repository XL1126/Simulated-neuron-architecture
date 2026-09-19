import random
from collections import deque


class WorkingMemory:
    def __init__(self, population, num_slots=7):
        self.pop = population
        self.num_slots = num_slots
        self.slots = [None] * num_slots
        self.slot_timestamps = [0] * num_slots
        self.slot_decay = [1.0] * num_slots
        self.decay_rate = 0.995
        self.read_noise = 0.05

        self.memory_neurons = []
        self._reserve_memory_neurons(200)

    def _reserve_memory_neurons(self, count):
        start = int(self.pop.size() * 0.7)
        for i in range(count):
            nid = start + (i * 17 + 3) % int(self.pop.size() * 0.2)
            if nid < self.pop.size():
                self.memory_neurons.append(nid)

    def update(self, step):
        fires = self.pop.get_current_fires()
        active_pattern = self._extract_pattern(fires)

        for i in range(self.num_slots):
            self.slot_decay[i] *= self.decay_rate

        oldest_slot = min(range(self.num_slots), key=lambda i: self.slot_decay[i])

        if active_pattern and len(active_pattern) > 3:
            self.slots[oldest_slot] = active_pattern
            self.slot_timestamps[oldest_slot] = step
            self.slot_decay[oldest_slot] = 1.0

        self._apply_lateral_inhibition()

    def _extract_pattern(self, fires):
        if len(fires) > 50:
            fires_sorted = sorted(fires, key=lambda f: f.strength, reverse=True)
            return fires_sorted[:20]
        return list(fires)

    def _apply_lateral_inhibition(self):
        for i in range(self.num_slots):
            for j in range(self.num_slots):
                if i != j and self.memory_neurons and i < len(self.memory_neurons) and j < len(self.memory_neurons):
                    src = self.memory_neurons[i % len(self.memory_neurons)]
                    dst = self.memory_neurons[j % len(self.memory_neurons)]
                    self.pop.add_synapse(src, dst, -0.1, 1)

    def read_slot(self, slot_index):
        if 0 <= slot_index < self.num_slots:
            pattern = self.slots[slot_index]
            if pattern:
                self.slot_decay[slot_index] *= 0.9
                return [self._add_read_noise(f) for f in pattern]
        return None

    def _add_read_noise(self, fire_event):
        noisy = type('SpikeFireEvent', (), {})()
        noisy.neuron_id = fire_event.neuron_id
        noisy.strength = fire_event.strength + random.gauss(0, self.read_noise)
        return noisy

    def similarity_to_slot(self, pattern, slot_index):
        if slot_index >= self.num_slots or self.slots[slot_index] is None:
            return 0.0

        stored = self.slots[slot_index]
        stored_ids = {f.neuron_id for f in stored}
        pattern_ids = {f.neuron_id for f in pattern}

        if not stored_ids or not pattern_ids:
            return 0.0

        intersection = stored_ids & pattern_ids
        union = stored_ids | pattern_ids
        return len(intersection) / len(union) if union else 0.0
