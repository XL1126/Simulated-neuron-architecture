import numpy as np
from collections import deque


class ImmediateMemory:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.tag_index = {}
        self.time_weights = {}
        self._overflow_count = 0
        self._overflow_warn_threshold = capacity // 2

    def push(self, event, tag=None):
        idx = len(self.buffer)
        self.buffer.append(event)
        self.time_weights[idx] = 1.0
        if tag:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(idx)
        self._decay_all(0.999)
        return idx

    def recall_by_tag(self, tag):
        if tag not in self.tag_index:
            return []
        indices = self.tag_index[tag]
        return [self.buffer[i] for i in indices if i < len(self.buffer)]

    def _decay_all(self, decay_rate):
        for idx in list(self.time_weights.keys()):
            if idx >= len(self.buffer):
                del self.time_weights[idx]
                continue
            self.time_weights[idx] *= decay_rate
            if self.time_weights[idx] < 0.01:
                del self.time_weights[idx]

    def get_weighted(self, indices):
        weighted = []
        for i in indices:
            if i < len(self.buffer):
                w = self.time_weights.get(i, 0.1)
                weighted.append((self.buffer[i], w))
        return weighted

    def clear(self):
        self.buffer.clear()
        self.tag_index.clear()
        self.time_weights.clear()


class ShortTermMemory:
    def __init__(self, capacity=7):
        self.slots = [None] * capacity
        self.slot_decay = [1.0] * capacity
        self.slot_tags = [None] * capacity
        self.decay_rate = 0.92
        self.replay_sequences = []
        self.next_write = 0

    def store(self, pattern, tag=None, strength=1.0):
        slot = self.next_write % len(self.slots)
        self.slots[slot] = np.array(pattern, copy=True)
        self.slot_decay[slot] = strength
        self.slot_tags[slot] = tag
        self.next_write = (self.next_write + 1) % len(self.slots)

    def recall(self, query_tag=None, top_k=1):
        results = []
        for i, (slot, tag) in enumerate(zip(self.slots, self.slot_tags)):
            if slot is not None and self.slot_decay[i] > 0.05:
                if query_tag is None or tag == query_tag:
                    results.append((slot.copy(), self.slot_decay[i], tag))

        results.sort(key=lambda x: x[1], reverse=True)
        for x in results:
            x[0] *= float(x[1])
        return [r[0] for r in results[:top_k]]

    def decay_step(self):
        for i in range(len(self.slot_decay)):
            self.slot_decay[i] *= self.decay_rate
            if self.slot_decay[i] < 0.05:
                self.slots[i] = None
                self.slot_tags[i] = None

    def save_replay_sequence(self, pattern):
        self.replay_sequences.append(np.array(pattern, copy=True))
        if len(self.replay_sequences) > 50:
            self.replay_sequences.pop(0)

    def get_random_replay(self):
        if not self.replay_sequences:
            return None
        return self.replay_sequences[np.random.randint(0, len(self.replay_sequences))]

    def pattern_completion(self, partial_pattern):
        best_match = None
        best_sim = -1
        for seq in self.replay_sequences:
            min_len = min(len(partial_pattern), len(seq))
            sim = np.dot(partial_pattern[:min_len], seq[:min_len]) / (np.linalg.norm(partial_pattern[:min_len]) * np.linalg.norm(seq[:min_len]) + 1e-10)
            if sim > best_sim:
                best_sim = sim
                best_match = seq
        return best_match, best_sim


class LongTermMemory:
    def __init__(self, consolidation_threshold=0.7):
        self.store = {}
        self.confidence = {}
        self.consolidation_threshold = consolidation_threshold
        self.consolidations = 0

    def consolidate(self, pattern_key, pattern_value, confidence):
        if confidence >= self.consolidation_threshold:
            existing = self.store.get(pattern_key)
            if existing is not None:
                self.store[pattern_key] = 0.8 * np.array(existing) + 0.2 * np.array(pattern_value)
            else:
                self.store[pattern_key] = np.array(pattern_value, copy=True)
                self.consolidations += 1
            self.confidence[pattern_key] = max(self.confidence.get(pattern_key, 0.0), confidence)

    def recall(self, pattern_key):
        if pattern_key in self.store:
            return self.store[pattern_key], self.confidence.get(pattern_key, 0.0)
        return None, 0.0

    def fuzzy_recall(self, query_vec, threshold=0.3):
        results = []
        for key, vec in self.store.items():
            sim = np.dot(query_vec.flatten(), vec.flatten()) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10)
            if sim > threshold:
                results.append((key, vec, sim))
        return sorted(results, key=lambda x: x[2], reverse=True)

    def downgrade_stale(self, decay=0.99):
        for key in list(self.confidence.keys()):
            self.confidence[key] *= decay
            if self.confidence[key] < 0.1:
                del self.store[key]
                del self.confidence[key]


class MemoryLayer:
    def __init__(self, population, config):
        self.pop = population
        self.total_neurons = population.size()

        self.instant = ImmediateMemory(capacity=config.get('instant_capacity', 100000))
        self.stm = ShortTermMemory(capacity=config.get('stm_capacity', 7))
        self.ltm = LongTermMemory(consolidation_threshold=config.get('consolidation_threshold', 0.6))

        self.pattern_vector_dim = config.get('pattern_dim', 128)
        self.total_consolidations = 0
        self.last_replay_vec = None

    def push_event(self, event, tag=None):
        return self.instant.push(event, tag)

    def store_pattern(self, pattern, tag=None, strength=1.0):
        self.stm.store(pattern, tag, strength)

    def recall_pattern(self, tag=None, top_k=1):
        return self.stm.recall(tag, top_k)

    def remember(self, query_tag):
        stm_result = self.stm.recall(query_tag, top_k=1)
        if stm_result:
            return stm_result[0], 'stm'

        ltm_result, confidence = self.ltm.recall(query_tag)
        if ltm_result is not None:
            return ltm_result, 'ltm'

        instant_result = self.instant.recall_by_tag(query_tag)
        if instant_result:
            return np.array(instant_result[-1]) if isinstance(instant_result[-1], (list, np.ndarray)) else instant_result[-1], 'instant'

        return None, 'none'

    def consolidate(self, pattern_key, pattern_value, confidence=0.7):
        self.ltm.consolidate(pattern_key, pattern_value, confidence)
        self.total_consolidations += 1

    def save_replay(self, pattern):
        self.stm.save_replay_sequence(pattern)

    def get_replay(self):
        return self.stm.get_random_replay()

    def step(self, step):
        self.stm.decay_step()
        if step % 1000 == 0:
            self.ltm.downgrade_stale(decay=0.995)

    def pattern_completion(self, partial):
        from_stm, stm_sim = self.stm.pattern_completion(partial)
        from_ltm = self.ltm.fuzzy_recall(np.array(partial), threshold=0.2)
        if from_ltm and from_ltm[0][2] > (stm_sim or 0):
            return from_ltm[0][1], 'ltm'
        if from_stm is not None and stm_sim > 0.3:
            return from_stm, 'stm'
        return None, 'none'

    def memory_stats(self):
        return {
            'instant_buffer': len(self.instant.buffer),
            'stm_slots': sum(1 for s in self.stm.slots if s is not None),
            'ltm_items': len(self.ltm.store),
            'consolidations': self.total_consolidations,
            'replay_sequences': len(self.stm.replay_sequences),
        }