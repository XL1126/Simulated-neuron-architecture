import numpy as np
from collections import defaultdict, deque


class WTAConvergenceDetector:
    def __init__(self, total_neurons, config):
        self.total_neurons = total_neurons
        self.decay = config.get('wta_decay', 0.95)
        self.inhibition_weight = config.get('wta_inhibition', 0.15)
        self.leader_ratio = config.get('wta_leader_ratio', 1.5)
        self.stable_time = config.get('wta_stable_time', 50)
        self.abs_threshold = config.get('wta_abs_threshold', 0.8)
        self.min_wait = config.get('convergence_min_wait', 80)
        self.max_wait = config.get('convergence_max_wait', 5000)

        self.candidates = np.zeros(total_neurons)
        self.leader_history = deque(maxlen=self.stable_time)
        self.current_leader = -1
        self.leader_count = 0
        self.last_fires = {}
        self.steps_since_start = 0
        self.converged = False
        self.converged_leader = -1
        self.conclusion_neurons = []
        self._init_inhibition_pools()

    def _init_inhibition_pools(self):
        self.inhibition_pools = {}
        pool_size = max(32, self.total_neurons // 8)
        for i in range(0, self.total_neurons, pool_size):
            pool_end = min(i + pool_size, self.total_neurons)
            self.inhibition_pools[i] = list(range(i, pool_end))

    def update(self, fires, step):
        self.steps_since_start = step

        current_fire_map = {}
        for f in fires:
            nid = f.neuron_id if hasattr(f, 'neuron_id') else f.get('neuron_id', 0)
            nid = nid % self.total_neurons
            strength = f.strength if hasattr(f, 'strength') else f.get('strength', 1.0)
            current_fire_map[nid] = current_fire_map.get(nid, 0) + strength

        self.candidates *= self.decay
        for nid, strength in current_fire_map.items():
            self.candidates[nid] += strength

        self._apply_lateral_inhibition()
        self._update_leader()
        self._check_convergence()

        self.last_fires = current_fire_map
        return self.converged

    def _apply_lateral_inhibition(self):
        for pool_start, members in self.inhibition_pools.items():
            total = sum(self.candidates[m] for m in members)
            if total > 0:
                for m in members:
                    inhibition = self.inhibition_weight * (total - self.candidates[m])
                    self.candidates[m] = max(0.0, self.candidates[m] - inhibition)

    def _update_leader(self):
        if self.candidates.max() < self.abs_threshold:
            self.leader_history.append(-1)
            self.current_leader = -1
            self.leader_count = 0
            return

        new_leader = int(np.argmax(self.candidates))
        sorted_vals = np.sort(self.candidates)[::-1]
        runner_up_val = sorted_vals[1] if len(sorted_vals) > 1 and sorted_vals[0] > 0 else 0.0

        leader_val = self.candidates[new_leader]
        if leader_val > 0 and leader_val >= runner_up_val * self.leader_ratio:
            self.leader_history.append(new_leader)
        else:
            self.leader_history.append(-1)

        if new_leader != self.current_leader:
            self.current_leader = new_leader
            self.leader_count = 1
        else:
            self.leader_count += 1

    def _check_convergence(self):
        if self.steps_since_start < self.min_wait:
            self.converged = False
            return

        if self.steps_since_start >= self.max_wait:
            self.converged = True
            self.converged_leader = self.current_leader if self.current_leader >= 0 else int(np.argmax(self.candidates))
            self.conclusion_neurons = self._gather_conclusion_neurons()
            return

        recent_leaders = list(self.leader_history)[-self.stable_time:]
        if len(recent_leaders) >= self.stable_time:
            same_leader = all(l == self.current_leader and l >= 0 for l in recent_leaders)
            if same_leader and self.candidates[self.current_leader] >= self.abs_threshold:
                self.converged = True
                self.converged_leader = self.current_leader
                self.conclusion_neurons = self._gather_conclusion_neurons()

    def _gather_conclusion_neurons(self):
        threshold = self.candidates.max() * 0.3
        active = np.where(self.candidates > threshold)[0]
        return list(active[:50])

    def reset(self):
        self.candidates = np.zeros(self.total_neurons)
        self.leader_history.clear()
        self.current_leader = -1
        self.leader_count = 0
        self.converged = False
        self.converged_leader = -1
        self.conclusion_neurons = []
        self.steps_since_start = 0
        self.last_fires = {}

    def get_candidate_activity(self):
        return {i: float(v) for i, v in enumerate(self.candidates) if v > 0.01}


class CoreLayer:
    def __init__(self, population, config, spa_bridge=None, mapper=None):
        self.pop = population
        self.total_neurons = population.size()
        self.spa_bridge = spa_bridge
        self.mapper = mapper

        self.wta = WTAConvergenceDetector(self.total_neurons, config)

        self.spontaneous_rate = config.get('spontaneous_rate', 0.002)
        self.conflict_threshold = config.get('conflict_threshold', 0.3)
        self.topology_built = False
        self.semantic_topology_built = False

        self.spontaneous_neurons = set()
        conflict_size = min(500, self.total_neurons // 4)
        for _ in range(conflict_size):
            self.spontaneous_neurons.add(np.random.randint(0, self.total_neurons))

        self.activity_history = deque(maxlen=100)
        self.conflict_events = deque(maxlen=20)
        self.last_thought_vector = np.zeros(64)

    def build_topology(self):
        if self.topology_built:
            return
        n = self.total_neurons
        self.pop.connect_random(0.005, 6)
        self.pop.build_small_world(50, 0.03, 0.005)
        self.pop.build_competitive_pool(int(n * 0.15), int(n * 0.45))
        self.pop.build_erdos_renyi(int(n * 0.5), int(n * 0.5) + min(300, n // 6), 0.03)
        self.topology_built = True

    def build_semantic_topology(self):
        if self.semantic_topology_built:
            return
        if not self.spa_bridge or not self.mapper:
            self.build_topology()
            self.semantic_topology_built = True
            return

        self.build_topology()

        core_concepts = [
            '我', '你', '是', '不', '有', '在', '和', '的', '了',
            '说', '看', '想', '知道', '可以', '好', '真', '对',
            '人', '动物', '水', '天', '地', '猫', '狗', '花', '树',
            '高兴', '难过', '喜欢', '_NOT_', '?',
        ]

        for i, c1 in enumerate(core_concepts):
            for c2 in core_concepts[i + 1:]:
                sim = self.spa_bridge.compute_semantic_connectivity(c1, c2)
                if sim > 0.08:
                    self.spa_bridge.wire_concepts(c1, c2, self.pop)

        self.semantic_topology_built = True
        return len(core_concepts) ** 2

    def step(self, fires, current_step):
        spike_count = len(fires)
        self.activity_history.append(spike_count)

        self.wta.update(fires, current_step)

        if self._detect_conflict(fires):
            self.conflict_events.append(current_step)
            self.pop.build_competitive_pool(
                int(self.total_neurons * 0.2), int(self.total_neurons * 0.5))

        self._update_thought_vector(fires)
        self._generate_spontaneous(current_step)

    def _detect_conflict(self, fires):
        if not fires:
            return False
        recent = list(self.activity_history)[-20:]
        if len(recent) < 10:
            return False
        variance = np.var(recent) / (np.mean(recent) + 1)
        return variance > self.conflict_threshold

    def _update_thought_vector(self, fires):
        if not fires:
            return
        thought = np.zeros(64)
        for f in fires[:32]:
            nid = f.neuron_id % 64
            thought[nid] += 1.0
        norm = np.linalg.norm(thought)
        if norm > 1e-10:
            thought /= norm
        self.last_thought_vector = 0.85 * self.last_thought_vector + 0.15 * thought

    def _generate_spontaneous(self, step):
        if np.random.random() < self.spontaneous_rate:
            nid = np.random.choice(list(self.spontaneous_neurons))
            self.pop.inject_spike(nid, np.random.uniform(0.5, 1.8), 1)

        self.pop.inject_spike_group(
            [np.random.randint(0, self.total_neurons) for _ in range(5)], 0.2, 1)

    def check_convergence(self):
        return self.wta.converged

    def get_conclusion(self):
        return {
            'leader': self.wta.converged_leader,
            'neurons': self.wta.conclusion_neurons,
            'activity': self.wta.get_candidate_activity(),
        }

    def get_thought_vector(self):
        return self.last_thought_vector

    def get_firing_summary(self):
        if not self.activity_history:
            return {'mean': 0, 'variance': 0}
        arr = np.array(self.activity_history)
        return {'mean': float(np.mean(arr)), 'variance': float(np.var(arr)), 'max': int(np.max(arr))}

    def reset_for_next_round(self):
        self.wta.reset()
        self.last_thought_vector = np.zeros(64)