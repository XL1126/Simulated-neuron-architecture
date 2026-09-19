import numpy as np
from collections import deque, defaultdict


class GlobalWorkspace:
    def __init__(self, dim=512, num_processors=5):
        self.dim = dim
        self.num_processors = num_processors

        self.workspace = np.zeros(dim)
        self.processor_buffers = {f'p{i}': np.zeros(dim) for i in range(num_processors)}
        self.processor_weights = {f'p{i}': np.random.normal(0, 0.1, (dim, dim))
                                  for i in range(num_processors)}
        self.processor_labels = {
            'p0': 'sensory_input', 'p1': 'semantic_core',
            'p2': 'memory_retrieval', 'p3': 'action_planning',
            'p4': 'self_reflection',
        }

        self.competition_strengths = {f'p{i}': 0.5 for i in range(num_processors)}
        self.broadcast_weights = np.random.normal(0, 0.05, (num_processors, dim))
        self.ignition_threshold = 0.4
        self.global_ignition = False
        self.ignition_duration = 0
        self.prev_workspace = np.zeros(dim)

        self.access_history = deque(maxlen=100)
        self.conscious_content = None
        self.consciousness_level = 0.0
        self.attention_spotlight = np.zeros(dim)
        self.phi_estimate = 0.0

        self.phase = 'rest'
        self.cycle_count = 0

    def write(self, processor_id, content, confidence=0.5):
        pid = f'p{processor_id}' if isinstance(processor_id, int) else processor_id
        content = np.array(content).flatten()
        if len(content) > self.dim:
            content = content[:self.dim]
        elif len(content) < self.dim:
            content = np.pad(content, (0, self.dim - len(content)))

        self.processor_buffers[pid] = content
        self.competition_strengths[pid] = confidence
        self.cycle_count += 1

    def compete(self):
        strengths = list(self.competition_strengths.items())
        strengths.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        readout_threshold = self.ignition_threshold
        for pid, strength in strengths:
            if strength >= readout_threshold:
                candidates.append((pid, strength, self.processor_buffers[pid]))

        if not candidates:
            self.global_ignition = False
            self.consciousness_level *= 0.95
            return None

        winner_pid, winner_strength, winner_content = candidates[0]
        self.workspace = 0.7 * winner_content + 0.3 * self.workspace
        workspace_norm = np.linalg.norm(self.workspace)
        if workspace_norm > 1e-10:
            self.workspace /= workspace_norm

        if len(candidates) > 1:
            runner_up = candidates[1][2]
            contrast = np.linalg.norm(winner_content - runner_up)
            if contrast > 0.3:
                self.consciousness_level = min(1.0,
                    self.consciousness_level + 0.1 + contrast * 0.5)
                self.global_ignition = True
                self.ignition_duration += 1
            else:
                self.global_ignition = False
                self.consciousness_level *= 0.9
        else:
            self.global_ignition = True
            self.ignition_duration += 1
            self.consciousness_level = min(1.0, self.consciousness_level + 0.08)

        self._compute_phi()
        self.conscious_content = winner_pid
        self.access_history.append({
            'winner': winner_pid, 'strength': winner_strength,
            'global_ignition': self.global_ignition, 'phi': self.phi_estimate,
        })

        return winner_pid, self.workspace.copy()

    def broadcast(self):
        broadcast_signal = self.workspace.copy()
        for pid in self.processor_buffers:
            idx = int(pid[1])
            modulation = np.dot(self.broadcast_weights[idx], broadcast_signal)
            self.processor_buffers[pid] += modulation * 0.3 * broadcast_signal

        self.attention_spotlight = 0.85 * self.attention_spotlight + 0.15 * broadcast_signal
        return broadcast_signal

    def cycle(self, processor_inputs):
        for pid, content, confidence in processor_inputs:
            self.write(pid, content, confidence)

        result = self.compete()
        if result:
            broadcast = self.broadcast()
            self.phase = 'conscious' if self.global_ignition else 'subconscious'
            return broadcast

        self.phase = 'rest'
        return None

    def _compute_phi(self):
        diff = np.linalg.norm(self.workspace - self.prev_workspace)
        integration = np.mean(np.abs(self.workspace)) * self.global_ignition
        differentiation = 0.0
        for pid_a in self.processor_buffers:
            for pid_b in self.processor_buffers:
                if pid_a < pid_b:
                    diff_ab = np.linalg.norm(
                        self.processor_buffers[pid_a] - self.processor_buffers[pid_b])
                    differentiation += diff_ab

        diff_cnt = max(1, len(self.processor_buffers) * (len(self.processor_buffers) - 1) / 2)
        differentiation /= diff_cnt

        self.phi_estimate = 0.3 * diff + 0.4 * integration + 0.3 * differentiation
        self.phi_estimate = max(0.0, min(1.0, self.phi_estimate))
        self.prev_workspace = self.workspace.copy()

    def get_conscious_content(self):
        if self.conscious_content and self.global_ignition:
            return {
                'processor': self.conscious_content,
                'content_vector': self.workspace.copy(),
                'consciousness_level': self.consciousness_level,
                'phi': self.phi_estimate,
                'phase': self.phase,
            }
        return None

    def get_workspace_summary(self):
        return {
            'phase': self.phase,
            'consciousness': round(self.consciousness_level, 3),
            'phi': round(self.phi_estimate, 3),
            'ignition': self.global_ignition,
            'cycles': self.cycle_count,
        }

    def reset(self):
        self.workspace = np.zeros(self.dim)
        self.global_ignition = False
        self.ignition_duration = 0
        self.conscious_content = None
        self.consciousness_level *= 0.5
        self.attention_spotlight = np.zeros(self.dim)
        self.phase = 'rest'