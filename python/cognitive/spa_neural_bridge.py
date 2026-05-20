import numpy as np
from collections import defaultdict, deque


class SPANeuralBridge:
    def __init__(self, spa, mapper, population, total_neurons):
        self.spa = spa
        self.mapper = mapper
        self.pop = population
        self.total_neurons = total_neurons

        self.semantic_weights = {}
        self.semantic_pairs = defaultdict(float)
        self.pair_access_count = defaultdict(int)
        self.learning_rate = 0.005
        self.similarity_threshold = 0.15
        self.max_weight = 0.3

        self.activation_history = deque(maxlen=200)
        self.bridge_stats = {'applications': 0, 'modulations': 0}

        self._core_concepts = ['我', '你', '是', '不', '有', '在', '和', '的', '了',
                               '问', '说', '看', '想', '知道', '可以', '会', '能',
                               '好', '大', '小', '多', '少', '真', '假', '对', '错',
                               '人', '动物', '植物', '水果', '水', '火', '天', '地',
                               '猫', '狗', '鸟', '鱼', '花', '树', '苹果', '香蕉',
                               '高兴', '难过', '愤怒', '害怕', '喜欢', '讨厌',
                               '_NOT_', '_POS_', '_EVENT_', '?', '!']
        for c in self._core_concepts:
            self.spa.get_vector(c)

    def compute_semantic_connectivity(self, concept_a, concept_b):
        if concept_a == concept_b:
            return self.max_weight
        return max(0.0, self.spa.similarity(
            self.spa.get_vector(concept_a),
            self.spa.get_vector(concept_b)))

    def compute_weight(self, src_neuron_ids, dst_neuron_ids, src_concept, dst_concept):
        pair_key = (src_concept, dst_concept)
        self.pair_access_count[pair_key] += 1

        base_sim = self.compute_semantic_connectivity(src_concept, dst_concept)
        learned = self.semantic_pairs.get(pair_key, 0.0)

        weight = base_sim * 0.6 + learned * 0.4
        weight = min(self.max_weight, max(0.001, weight))

        if base_sim > self.similarity_threshold:
            weight += 0.02

        return weight

    def wire_concepts(self, src_concept, dst_concept, population_obj=None):
        pop = population_obj or self.pop
        src_nids = self.mapper.allocate(src_concept)
        dst_nids = self.mapper.allocate(dst_concept)

        base_weight = self.compute_weight(src_nids, dst_nids, src_concept, dst_concept)
        connections = 0

        for s in src_nids[:4]:
            for d in dst_nids[:4]:
                if s != d:
                    w = base_weight * np.random.uniform(0.8, 1.2)
                    dly = max(1, int(abs(self.spa.similarity(
                        self.spa.get_vector(src_concept),
                        self.spa.get_vector(dst_concept))) * 5 + 1))
                    pop.add_synapse(s % self.total_neurons, d % self.total_neurons, w, dly)
                    connections += 1

        self.bridge_stats['applications'] += 1
        return connections

    def wire_vocabulary(self, concepts, population_obj=None):
        pop = population_obj or self.pop
        total = 0
        concepts = list(set(concepts))[:200]

        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                sim = self.compute_semantic_connectivity(c1, c2)
                if sim > 0.10:
                    total += self.wire_concepts(c1, c2, pop)

        for c in concepts[:100]:
            for core in self._core_concepts[:50]:
                sim = self.compute_semantic_connectivity(c, core)
                if sim > 0.08:
                    total += self.wire_concepts(c, core, pop)

        return total

    def reinforce_pair(self, concept_a, concept_b, reward, alpha=0.01):
        pair_key = (concept_a, concept_b)
        self.semantic_pairs[pair_key] += reward * alpha
        self.semantic_pairs[pair_key] = max(0.0, min(self.max_weight,
            self.semantic_pairs[pair_key]))
        self.bridge_stats['modulations'] += 1

    def reinforce_from_fires(self, fires, reward, top_k=10):
        if not fires:
            return

        concepts_from_fires = set()
        for f in fires[:top_k]:
            nid = f.neuron_id % self.total_neurons
            mapped = self.mapper.concepts_for_neuron(nid)
            concepts_from_fires.update(mapped)

        for c in list(concepts_from_fires)[:5]:
            key = (c, c)
            self.semantic_pairs[key] += reward * 0.005
            self.semantic_pairs[key] = min(self.max_weight,
                self.semantic_pairs[key])

    def modulate_connection(self, concept_a, concept_b, delta):
        pair_key = (concept_a, concept_b)
        self.semantic_pairs[pair_key] += delta
        self.semantic_pairs[pair_key] = max(-0.5, min(0.5,
            self.semantic_pairs[pair_key]))

    def get_bridge_summary(self):
        return {
            'pair_count': len(self.semantic_pairs),
            'applications': self.bridge_stats['applications'],
            'modulations': self.bridge_stats['modulations'],
            'top_pairs': sorted(self.semantic_pairs.items(),
                                key=lambda x: x[1], reverse=True)[:10],
        }

    def decay_all(self, rate=0.999):
        for key in list(self.semantic_pairs.keys()):
            self.semantic_pairs[key] *= rate
            if abs(self.semantic_pairs[key]) < 0.001:
                del self.semantic_pairs[key]