import numpy as np
from collections import deque
from ..cognitive.semantic_pointer import SemanticPointer, DistributedConceptMapper


class InputLayer:
    def __init__(self, population, config):
        self.pop = population
        total = population.size()
        self.total_neurons = total

        self.dim = config.get('spa_dim', 512)
        self.neurons_per_concept = config.get('neurons_per_concept', 12)
        self.composition_depth = config.get('composition_depth', 3)
        self.use_composition = config.get('use_composition', True)

        self.spa = SemanticPointer(dim=self.dim)
        self.mapper = DistributedConceptMapper(
            self.spa, self.total_neurons, self.neurons_per_concept
        )

        self.char_set = self._build_char_set()
        for ch in self.char_set:
            self.mapper.allocate(ch)

        self.recent_inputs = deque(maxlen=10)
        self.context_window = deque(maxlen=config.get('context_window', 6))
        self.context_vector = np.zeros(self.dim)

        self.learned_patterns = {}
        self.pattern_importance = {}

    def _build_char_set(self):
        chars = set()
        chars.update(chr(i) for i in range(0x4E00, 0x9FFF))
        chars.update(chr(i) for i in range(0x3040, 0x309F))
        chars.update(chr(i) for i in range(0x30A0, 0x30FF))
        for i in range(32, 127):
            chars.add(chr(i))
        chars.update(['\n', '\r', '\t'])
        return chars

    def encode_char(self, char, strength=1.0):
        concept_neurons = self.mapper.allocate(char)
        events = []
        for nid in concept_neurons:
            phase = np.random.uniform(0.0, 0.1)
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': strength * 0.8,
                'phase': phase,
                'source': 'input_char',
                'char': char,
            })
        seed_vec = self.spa.get_vector(char)
        self.context_window.append((char, seed_vec))
        self._update_context_vector()

        compositional_neurons = self._encode_compositional(char)
        for nid in compositional_neurons:
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': strength * 0.35,
                'phase': 0.0,
                'source': 'compositional',
            })

        return events

    def encode_word(self, word, strength=1.0):
        events = []
        word_vec = self.spa.get_vector(word)
        word_neurons = self.mapper.neurons_for_vector(word_vec)
        for nid in word_neurons:
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': strength,
                'phase': 0.0,
                'source': 'word',
                'word': word,
            })

        for ch in word:
            char_nids = self.mapper.allocate(ch)
            for nid in char_nids:
                events.append({
                    'neuron_id': nid % self.total_neurons,
                    'strength': strength * 0.3,
                    'phase': np.random.uniform(0.0, 0.05),
                    'source': 'char_in_word',
                })

        self.recent_inputs.append(word)
        return events

    def encode_text(self, text, strength=1.0):
        events = []
        words = text.split()
        for word in words:
            events.extend(self.encode_word(word, strength))
        return events

    def encode_negation(self, text):
        events = self.encode_text(text, 0.7)
        not_kids = self.mapper.allocate('_NOT_')
        for nid in not_kids:
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': 1.2,
                'phase': 0.0,
                'source': 'negation',
            })
        return events

    def encode_question(self, text):
        events = self.encode_text(text, 0.9)
        q_nids = self.mapper.allocate('?')
        for nid in q_nids:
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': 0.8,
                'phase': 0.0,
                'source': 'question_mark',
            })
        return events

    def _encode_compositional(self, char):
        neuron_ids = []
        if self.context_window and len(self.context_window) >= 2:
            recent_chars = [c for c, _ in list(self.context_window)[-2:]]
            compound_str = ''.join(recent_chars) + char
            if len(compound_str) > 1:
                compound_vec = self.spa.compose_phrase(list(compound_str))
                compound_nids = self.mapper.neurons_for_vector(compound_vec)
                neuron_ids.extend(compound_nids[:4])

                if compound_str not in self.learned_patterns:
                    self.learned_patterns[compound_str] = compound_vec
                    self.pattern_importance[compound_str] = 0.1
        return neuron_ids

    def _update_context_vector(self):
        if not self.context_window:
            self.context_vector = np.zeros(self.dim)
            return
        vecs = [v for _, v in self.context_window]
        self.context_vector = self.spa.superpose(vecs)

    def get_context_vector(self):
        return self.context_vector

    def fuzzy_lookup(self, vector, threshold=0.15):
        results = []
        for pattern, (char, vec) in self.context_window.__dict__.get('_data', {}).items():
            sim = self.spa.similarity(vector, vec)
            if sim > threshold:
                results.append((char, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def learn_pattern(self, pattern_str):
        vec = self.spa.compose_phrase(list(pattern_str))
        self.learned_patterns[pattern_str] = vec
        self.pattern_importance[pattern_str] = 0.2

    def reinforce_pattern(self, pattern_str, delta=0.05):
        if pattern_str in self.pattern_importance:
            self.pattern_importance[pattern_str] = min(1.0,
                self.pattern_importance[pattern_str] + delta)

    def get_semantic_similarity(self, word_a, word_b):
        va = self.spa.get_vector(word_a)
        vb = self.spa.get_vector(word_b)
        return self.spa.similarity(va, vb)

    def encode_embodied_event(self, event_type, data, strength=0.6):
        events = []
        event_nids = self.mapper.allocate('_EVENT_' + event_type)
        for nid in event_nids:
            events.append({
                'neuron_id': nid % self.total_neurons,
                'strength': strength,
                'phase': 0.0,
                'source': 'embodied_event',
                'event_type': event_type,
                'data': str(data),
            })

        if isinstance(data, str):
            data_events = self.encode_text(data, strength * 0.5)
            events.extend(data_events)

        return events