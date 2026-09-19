import numpy as np
from collections import deque, defaultdict
from python.utils.tensor_utils import to_fixed_dim, normalize, cosine_sim


class CausalNode:
    def __init__(self, node_id, vector=None, dim=256):
        self.id = node_id
        self.vector = normalize(to_fixed_dim(vector, dim)) if vector is not None \
            else normalize(np.random.normal(0, 0.1, dim))
        self.activity = 0.5
        self.prediction_error = 0.0
        self.cumulative_error = 0.0

    def update_activity(self, new_vec, alpha=0.3):
        new_vec = to_fixed_dim(new_vec, len(self.vector))
        self.vector = (1 - alpha) * self.vector + alpha * new_vec
        self.vector = normalize(self.vector)


class CausalEdge:
    def __init__(self, src_id, dst_id, delay=1):
        self.src_id = src_id
        self.dst_id = dst_id
        self.delay = delay
        self.weight = 0.05
        self.eligibility = 0.1
        self.use_count = 0
        self.is_potentiated = False

    def update_weight(self, pre_activity, post_activity, reward=0.0):
        dw = 0.001 * pre_activity * post_activity + reward * 0.005
        self.weight += dw
        self.weight = max(-0.5, min(1.0, self.weight))
        self.eligibility = 0.9 * self.eligibility + 0.1 * abs(dw)
        self.use_count += 1
        if abs(self.weight) > 0.3:
            self.is_potentiated = True


class CausalGraph:
    def __init__(self, spa, dim=256, max_nodes=200):
        self.spa = spa
        self.dim = dim
        self.max_nodes = max_nodes

        self.nodes = {}
        self.edges = {}
        self.node_counter = 0

        self.free_energy = 0.5
        self.risk = 0.3
        self.ambiguity = 0.3
        self.precision = np.ones(dim, dtype=np.float32)
        self.prediction_history = deque(maxlen=50)

    def add_node(self, concept, vector=None):
        if concept in self.nodes:
            return self.nodes[concept].id

        if len(self.nodes) >= self.max_nodes:
            self._prune_least_used()

        node = CausalNode(self.node_counter, vector, self.dim)
        self.nodes[concept] = node
        self.node_counter += 1
        return node.id

    def add_edge(self, concept_a, concept_b, delay=1):
        if concept_a not in self.nodes:
            self.add_node(concept_a)
        if concept_b not in self.nodes:
            self.add_node(concept_b)

        node_a = self.nodes[concept_a]
        node_b = self.nodes[concept_b]
        key = (node_a.id, node_b.id)

        if key not in self.edges:
            self.edges[key] = CausalEdge(node_a.id, node_b.id, delay)

        return self.edges[key]

    def predict(self, active_concepts, horizon=3):
        predictions = []
        current = {}

        for concept in active_concepts:
            if concept in self.nodes:
                node = self.nodes[concept]
                current[node.id] = node.vector.copy()

        for h in range(horizon):
            next_state = {}
            for (src, dst), edge in self.edges.items():
                if src in current and edge.delay <= h + 1:
                    src_vec = current[src]
                    if dst not in next_state:
                        next_state[dst] = np.zeros(self.dim, dtype=np.float32)
                    next_state[dst] += edge.weight * src_vec * edge.eligibility

            for nid, vec in next_state.items():
                next_state[nid] = normalize(vec)

            predictions.append(next_state)
            current = next_state

        return predictions

    def compute_free_energy(self, expected_state, observed_state):
        expected = to_fixed_dim(expected_state, self.dim)
        observed = to_fixed_dim(observed_state, self.dim)

        prediction_error = expected - observed
        self.risk = 0.5 * np.sum(prediction_error ** 2 * self.precision)
        self.ambiguity = 0.5 * np.sum(1.0 / (self.precision + 1e-10))
        self.free_energy = self.risk + self.ambiguity

        self._update_precision(prediction_error)
        self.prediction_history.append(float(self.free_energy))
        return self.free_energy

    def learn_from_transition(self, pre_concepts, post_concepts, reward=0.0):
        for pre in pre_concepts:
            for post in post_concepts:
                edge = self.add_edge(pre, post)
                pre_activity = self.nodes[pre].activity if pre in self.nodes else 0.5
                post_activity = self.nodes[post].activity if post in self.nodes else 0.5
                edge.update_weight(pre_activity, post_activity, reward)

        for c in pre_concepts:
            if c in self.nodes:
                self.nodes[c].activity = min(1.0, self.nodes[c].activity + 0.05)

        for c in (set(pre_concepts) | set(post_concepts)):
            if c in self.nodes:
                self.nodes[c].update_activity(
                    np.zeros(self.dim) if c not in post_concepts
                    else np.ones(self.dim) * 0.1)

    def get_top_causes(self, effect_concept, top_k=5):
        results = []
        if effect_concept not in self.nodes:
            return results
        effect_id = self.nodes[effect_concept].id

        for (src, dst), edge in self.edges.items():
            if dst == effect_id and edge.is_potentiated:
                src_concept = self._id_to_concept(src)
                if src_concept:
                    results.append((src_concept, edge.weight, edge.eligibility))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results[:top_k]

    def get_summary(self):
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'potentiated_edges': sum(1 for e in self.edges.values() if e.is_potentiated),
            'free_energy': round(self.free_energy, 4),
            'risk': round(self.risk, 4),
            'ambiguity': round(self.ambiguity, 4),
            'precision_mean': round(float(np.mean(self.precision)), 4),
        }

    def _id_to_concept(self, node_id):
        for concept, node in self.nodes.items():
            if node.id == node_id:
                return concept
        return None

    def _prune_least_used(self):
        if not self.edges:
            return
        unused = sorted(self.edges.items(), key=lambda x: x[1].use_count)
        for key, _ in unused[:max(1, len(unused) // 4)]:
            del self.edges[key]

    def _update_precision(self, prediction_error):
        self.precision = 0.95 * self.precision + 0.05 * (prediction_error ** 2 + 1.0)
        self.precision = np.clip(self.precision, 0.1, 10.0)