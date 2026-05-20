import random
import numpy as np
from collections import defaultdict


class PrimaryLayer:
    def __init__(self, population, num_clusters=256, cluster_neurons_per_input=10):
        self.pop = population
        self.num_clusters = num_clusters
        self.cluster_neurons_per_input = cluster_neurons_per_input
        total_primary = int(population.size() * 0.3)
        self.primary_start = 0
        self.primary_end = total_primary

        self.clusters = {}
        for i in range(total_primary):
            cluster_id = i % num_clusters
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

        self.last_fired = {i: 0 for i in range(total_primary)}
        self.cooccurrence = defaultdict(int)
        self.cluster_cooccurrence = defaultdict(int)
        self.neuron_importance = defaultdict(float)

        self._build_initial_topology()

    def _build_initial_topology(self):
        for cluster_id, neuron_ids in self.clusters.items():
            num_neurons = len(neuron_ids)
            for i in range(num_neurons):
                src = neuron_ids[i]
                for offset in range(1, min(10, num_neurons - i)):
                    dst = neuron_ids[i + offset]
                    if src != dst:
                        w = random.uniform(0.05, 0.15)
                        self.pop.add_synapse(src, dst, w, random.randint(1, 3))
                        self.pop.add_synapse(dst, src, w, random.randint(1, 3))

        cluster_ids = list(self.clusters.keys())
        for c in cluster_ids:
            other = random.choice(cluster_ids)
            if other != c:
                src = random.choice(self.clusters[c])
                dst = random.choice(self.clusters[other])
                self.pop.add_synapse(
                    src, dst,
                    random.uniform(0.005, 0.05),
                    random.randint(1, 10))

    def inject_spike(self, spike_info):
        strength = spike_info.get('strength', 1.0)
        group_hint = spike_info.get('group_hint', 0)
        target_id = spike_info.get('target_id', -1)
        time_ms = spike_info.get('time_ms', 0)

        if target_id >= 0:
            cluster_id = target_id % self.num_clusters
        else:
            cluster_id = group_hint % self.num_clusters

        if cluster_id in self.clusters:
            candidates = sorted(
                self.clusters[cluster_id],
                key=lambda nid: (self.last_fired.get(nid, 0), -self.neuron_importance.get(nid, 0))
            )

            num_to_activate = min(self.cluster_neurons_per_input, len(candidates))
            for i in range(num_to_activate):
                nid = candidates[i]
                injected_strength = strength * (1.0 / (i + 1))
                self.pop.inject_spike(nid, injected_strength, 1)
                self.last_fired[nid] = time_ms
                self.neuron_importance[nid] += 0.01 * strength

            lateral_count = max(1, num_to_activate // 3)
            for i in range(lateral_count):
                lateral_cluster = (cluster_id + 1 + i * 3) % self.num_clusters
                if lateral_cluster in self.clusters:
                    lateral_nid = random.choice(self.clusters[lateral_cluster])
                    self.pop.inject_spike(lateral_nid, strength * 0.15, 1)
                    self.last_fired[lateral_nid] = time_ms

    def inject_spikes(self, spike_list):
        for s in spike_list:
            self.inject_spike(s)

    def update_cooccurrence(self, active_neurons, step):
        for i, n1 in enumerate(active_neurons):
            for n2 in active_neurons[i + 1:]:
                if n1 != n2:
                    pair = (min(n1, n2), max(n1, n2))
                    self.cooccurrence[pair] += 1

                    c1 = n1 % self.num_clusters
                    c2 = n2 % self.num_clusters
                    if c1 != c2:
                        c_pair = (min(c1, c2), max(c1, c2))
                        self.cluster_cooccurrence[c_pair] += 1

    def form_bridge_synapses(self, threshold=5):
        formed = 0
        for (c1, c2), count in self.cluster_cooccurrence.items():
            if count > threshold:
                if c1 in self.clusters and c2 in self.clusters:
                    src = random.choice(self.clusters[c1])
                    dst = random.choice(self.clusters[c2])
                    self.pop.add_synapse(src, dst, 0.08, random.randint(1, 5))
                    formed += 1

        for key in list(self.cluster_cooccurrence.keys()):
            self.cluster_cooccurrence[key] = max(0, self.cluster_cooccurrence[key] - 1)

        return formed

    def get_cluster_size(self, cluster_id):
        return len(self.clusters.get(cluster_id, []))

    def get_cluster_activity(self, fires):
        activity = defaultdict(float)
        for f in fires:
            cluster_id = f.neuron_id % self.num_clusters
            activity[cluster_id] += f.strength
        return dict(activity)

    def reinforce_cluster(self, cluster_id, strength=0.5):
        if cluster_id in self.clusters:
            for nid in self.clusters[cluster_id][:5]:
                self.pop.inject_spike(nid, strength, 1)