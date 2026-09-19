import numpy as np
import math
from collections import defaultdict


class KuramotoOscillator:
    def __init__(self, natural_freq, coupling=1.0):
        self.phase = np.random.uniform(0, 2 * math.pi)
        self.natural_freq = natural_freq
        self.coupling = coupling
        self.amplitude = 1.0

    def step(self, dt, neighbors_phases):
        d_phase = self.natural_freq
        sync_term = 0.0
        for neighbor_phase in neighbors_phases:
            sync_term += self.coupling * math.sin(neighbor_phase - self.phase)
        if neighbors_phases:
            sync_term /= len(neighbors_phases)
        d_phase += sync_term
        self.phase += d_phase * dt
        self.phase %= 2 * math.pi
        return self.phase


class OscillatoryBinding:
    def __init__(self, num_neurons, gamma_freq=40.0, theta_freq=5.0):
        self.num_neurons = num_neurons
        self.gamma_freq = gamma_freq
        self.theta_freq = theta_freq

        self.oscillators = {}
        self.binding_groups = defaultdict(list)
        self.sync_order = defaultdict(float)
        self.global_coherence = 0.0

    def register_neurons(self, neuron_ids, freq_variation=2.0):
        for nid in neuron_ids:
            if nid not in self.oscillators:
                base_freq = self.gamma_freq + np.random.normal(0, freq_variation)
                self.oscillators[nid] = KuramotoOscillator(
                    natural_freq=base_freq * 2 * math.pi / 1000.0,
                    coupling=np.random.uniform(0.8, 1.2)
                )

    def bind_group(self, group_name, neuron_ids):
        self.register_neurons(neuron_ids)
        self.binding_groups[group_name] = list(neuron_ids)

    def step(self, dt=1.0, active_neurons=None):
        if active_neurons is None:
            active_neurons = set()

        phase_map = {}
        for nid, osc in self.oscillators.items():
            neighbors = []
            for group_name, members in self.binding_groups.items():
                if nid in members:
                    for m in members:
                        if m != nid and m in active_neurons:
                            neighbors.append(self.oscillators[m].phase)
            phase_map[nid] = osc.step(dt, neighbors)

        self._compute_synchrony(active_neurons)
        return phase_map

    def _compute_synchrony(self, active_neurons):
        if not active_neurons or len(active_neurons) < 2:
            self.global_coherence = 0.0
            return

        for group_name, members in self.binding_groups.items():
            active_members = [m for m in members if m in active_neurons]
            if len(active_members) < 2:
                self.sync_order[group_name] = 0.0
                continue

            phases = [self.oscillators[m].phase for m in active_members]
            r = abs(np.mean([math.cos(p) + 1j * math.sin(p) for p in phases]))
            self.sync_order[group_name] = r

        active_with_groups = set()
        for group_name, members in self.binding_groups.items():
            active_with_groups.update(m for m in members if m in active_neurons)

        if active_with_groups:
            phases = [self.oscillators[n].phase for n in active_with_groups]
            self.global_coherence = abs(
                np.mean([math.cos(p) + 1j * math.sin(p) for p in phases]))

    def get_coherent_groups(self, threshold=0.5):
        return {g: s for g, s in self.sync_order.items() if s >= threshold}

    def is_bound(self, group_name):
        return self.sync_order.get(group_name, 0.0) > 0.5

    def phase_similarity(self, nid_a, nid_b):
        if nid_a not in self.oscillators or nid_b not in self.oscillators:
            return 0.0
        pa = self.oscillators[nid_a].phase
        pb = self.oscillators[nid_b].phase
        return math.cos(pa - pb)

    def get_modulation_factor(self, neuron_id):
        osc = self.oscillators.get(neuron_id)
        if osc is None:
            return 1.0
        gamma_phase = math.sin(osc.phase)
        return 0.7 + 0.3 * gamma_phase