import numpy as np
from collections import defaultdict

class SemanticPointer:
    def __init__(self, dim=512):
        self.dim = dim
        self.vocabulary = {}
        self.relations = {}

    def generate(self, seed=None):
        rng = np.random.RandomState(seed) if seed else np.random
        vec = rng.normal(0, 1.0 / np.sqrt(self.dim), self.dim)
        return vec / np.linalg.norm(vec)

    def add_symbol(self, symbol, seed=None):
        if symbol not in self.vocabulary:
            self.vocabulary[symbol] = self.generate(seed=seed)
        return self.vocabulary[symbol]

    def get_vector(self, symbol):
        if symbol not in self.vocabulary:
            self.add_symbol(symbol, seed=hash(symbol) % (2**31))
        return self.vocabulary[symbol]

    def bind(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        result = np.fft.irfft(fa * fb, n=len(a))
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result.astype(np.float64)

    def unbind(self, bound, key):
        key = np.asarray(key, dtype=np.float64)
        n = len(key)
        inv_key = np.empty(n, dtype=np.float64)
        inv_key[0] = key[0]
        inv_key[1:] = key[:0:-1]
        return self.bind(bound, inv_key)

    def superpose(self, vectors, weights=None):
        if not vectors:
            return np.zeros(self.dim)
        result = np.zeros(self.dim)
        for i, v in enumerate(vectors):
            w = weights[i] if weights else 1.0 / len(vectors)
            result += w * np.array(v)
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result

    def similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)

    def clean_up(self, noisy_vec, threshold=0.3):
        best_symbol = None
        best_sim = threshold
        for symbol, vec in self.vocabulary.items():
            sim = self.similarity(noisy_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_symbol = symbol
        return best_symbol, best_sim

    def compose_phrase(self, words):
        if not words:
            return np.zeros(self.dim)
        result = self.get_vector(words[0]).copy()
        for w in words[1:]:
            result = self.bind(result, self.get_vector(w))
        return result

    def encode_sequence(self, words):
        result = np.zeros(self.dim)
        for i, w in enumerate(words):
            pos_vec = self.get_vector(f"_POS_{i}")
            word_vec = self.get_vector(w)
            bound = self.bind(pos_vec, word_vec)
            result += bound
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result

    def add_relation(self, rel_type, a, b):
        if rel_type not in self.relations:
            self.relations[rel_type] = self.generate(seed=hash(rel_type) % (2**31))
        rel_vec = self.relations[rel_type]
        role_vec = self.get_vector(a)
        filler_vec = self.get_vector(b)
        return self.bind(rel_vec, self.superpose([role_vec, filler_vec]))

    def encode_negation(self, symbol):
        not_vec = self.get_vector("_NOT_")
        return self.bind(not_vec, self.get_vector(symbol))

    def decode_negation_similarity(self, vector, symbol):
        not_vec = self.get_vector("_NOT_")
        negated = self.bind(not_vec, self.get_vector(symbol))
        return self.similarity(vector, negated)

    def resolve_analogy(self, a_relation_b, c, top_k=5):
        items = sorted(self.vocabulary.items(),
                       key=lambda x: self.similarity(
                           self.unbind(a_relation_b, self.get_vector(x[0])),
                           self.get_vector(c)),
                       reverse=True)
        return [(sym, sim) for sym, sim in items[:top_k] if not sym.startswith('_')]

    def decay_vector(self, vec, rate=0.99):
        return vec * rate


class DistributedConceptMapper:
    def __init__(self, spa, total_neurons, neurons_per_concept=8):
        self.spa = spa
        self.total_neurons = total_neurons
        self.neurons_per_concept = neurons_per_concept
        self.concept_to_neurons = {}
        self.neuron_to_concepts = defaultdict(list)
        self.next_base_id = 0

    def allocate(self, concept):
        if concept in self.concept_to_neurons:
            return self.concept_to_neurons[concept]

        vec = self.spa.get_vector(concept)
        neuron_ids = []
        base = self.next_base_id
        for i in range(self.neurons_per_concept):
            segment = vec[i * len(vec) // self.neurons_per_concept:
                           (i + 1) * len(vec) // self.neurons_per_concept]
            nid = (base + int(abs(np.sum(segment)) * 1000)) % self.total_neurons
            neuron_ids.append(nid)
            self.neuron_to_concepts[nid].append(concept)
        self.next_base_id += self.neurons_per_concept
        self.concept_to_neurons[concept] = neuron_ids
        return neuron_ids

    def neurons_for_vector(self, vector):
        base = hash(str(vector.tobytes()[:32])) % self.total_neurons
        neuron_ids = []
        for i in range(self.neurons_per_concept):
            segment = vector[i * len(vector) // self.neurons_per_concept:
                             (i + 1) * len(vector) // self.neurons_per_concept]
            nid = (base + int(abs(np.sum(segment)) * 1000) + i * 7) % self.total_neurons
            neuron_ids.append(nid)
        return neuron_ids

    def concepts_for_neuron(self, neuron_id):
        return self.neuron_to_concepts.get(neuron_id, [])

    def vector_from_neuron_ids(self, neuron_ids):
        result = np.zeros(self.spa.dim)
        for nid in neuron_ids:
            concepts = self.neuron_to_concepts.get(nid, [])
            for c in concepts:
                result += self.spa.get_vector(c)
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result

    def encode_composition(self, components):
        compound = self.spa.compose_phrase(components)
        return self.neurons_for_vector(compound), compound