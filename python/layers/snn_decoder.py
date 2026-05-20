import numpy as np
from collections import deque, defaultdict


class SNNDecoder:
    def __init__(self, config, dtype=np.float32):
        self.num_neurons = config.get('decoder_neurons', 1000)
        self.char_vocab_size = config.get('char_vocab_size', 12000)
        self._dtype = dtype

        self.v = np.random.normal(-65.0, 5.0, self.num_neurons).astype(dtype)
        self.u = np.zeros(self.num_neurons, dtype=dtype)
        self.I = np.zeros(self.num_neurons, dtype=dtype)
        self.threshold = 30.0
        self.reset_v = dtype(-65.0)
        self.reset_u = dtype(2.0)
        self.a = dtype(0.02)
        self.b = dtype(0.2)

        self.input_weights = np.random.normal(0, 0.05, (self.num_neurons, 256)).astype(dtype)
        self.recurrent_weights = np.random.normal(0, 0.02,
            (self.num_neurons, self.num_neurons)).astype(dtype)
        self.output_weights = np.random.normal(0, 0.01,
            (self.char_vocab_size, self.num_neurons)).astype(dtype)

        self.output_activity = np.ones(self.char_vocab_size, dtype=dtype) * dtype(0.01)
        self.char_map = {}
        self.reverse_char_map = {}
        for i in range(32, 127):
            self.char_map[chr(i)] = i % self.char_vocab_size
            self.reverse_char_map[i % self.char_vocab_size] = chr(i)
        for i, ch in enumerate(['\n', '\r', '\t', ' ', '\uff0c', '\u3002', '\uff01', '\uff1f',
                                 '\u2026', '\u3001']):
            idx = (1000 + i) % self.char_vocab_size
            self.char_map[ch] = idx
            self.reverse_char_map[idx] = ch
        self.default_char = '\u2026'

        self.output_window = deque(maxlen=20)
        self.last_output_char = None
        self.output_counter = {}
        self._eligibility = defaultdict(float)
        self._decay_counter = 0

    def reset_state(self):
        self.v = np.random.normal(-65.0, 3.0, self.num_neurons).astype(self._dtype)
        self.u = np.zeros(self.num_neurons, dtype=self._dtype)
        self.I = np.zeros(self.num_neurons, dtype=self._dtype)
        self.output_activity = np.ones(self.char_vocab_size, dtype=self._dtype) * self._dtype(0.01)
        self.output_window.clear()
        self.last_output_char = None
        self.output_counter = {}
        self._eligibility.clear()
        self._decay_counter = 0

    def _to_fixed_input(self, input_vector):
        x = np.array(input_vector, dtype=self._dtype).flatten()
        if len(x) > 256:
            return x[:256]
        if len(x) < 256:
            return np.pad(x, (0, 256 - len(x)))
        return x

    def forward(self, input_vector, fires=None):
        input_vec = self._to_fixed_input(input_vector)
        self.I = self.input_weights @ input_vec * self._dtype(5.0)
        self.I += np.random.normal(0, 0.3, self.num_neurons).astype(self._dtype)
        self.I += self.recurrent_weights @ (self.v > -40.0).astype(self._dtype) * self._dtype(2.0)

        dt = self._dtype(0.5)
        self.v += dt * (self._dtype(0.04) * self.v ** 2 + self._dtype(5.0) * self.v
                         + self._dtype(140.0) - self.u + self.I)
        self.u += dt * self.a * (self.b * self.v - self.u)

        spike_mask = self.v >= self.threshold
        spike_indices = np.where(spike_mask)[0]
        fired_vec = spike_mask.astype(self._dtype) * self._dtype(20.0)
        self.output_activity = self._dtype(0.85) * self.output_activity + self._dtype(0.15) * (
            self.output_weights @ fired_vec)

        self.v[spike_mask] = self.reset_v
        self.u[spike_mask] += self.reset_u

        self._update_sparse_eligibility(spike_indices)

        return self._decode_output()

    def _update_sparse_eligibility(self, spike_indices):
        for key in list(self._eligibility.keys()):
            self._eligibility[key] *= 0.9

        if len(spike_indices) == 0:
            return

        active_out = np.where(self.output_activity > 0.005)[0]
        if len(active_out) > 50:
            active_out = active_out[np.argsort(self.output_activity[active_out])[::-1][:50]]

        for out_idx in active_out:
            out_idx = int(out_idx)
            for nidx in spike_indices:
                nidx = int(nidx)
                if nidx >= self.num_neurons:
                    continue
                w = float(self.output_weights[out_idx, nidx])
                if abs(w) > 1e-5:
                    key = (out_idx, nidx)
                    self._eligibility[key] += 0.1 * w

        self._decay_counter += 1
        if self._decay_counter > 1000:
            self._prune_eligibility()

    def _prune_eligibility(self):
        self._decay_counter = 0
        dead = [k for k, v in self._eligibility.items() if abs(v) < 0.0001]
        for k in dead:
            del self._eligibility[k]

    def _decode_output(self):
        top_idx = int(np.argmax(self.output_activity))
        top_val = float(self.output_activity[top_idx])
        self.output_window.append(top_idx)
        self.output_counter[top_idx] = self.output_counter.get(top_idx, 0) + 1
        char = self.reverse_char_map.get(top_idx, self.default_char)
        confidence = top_val / (float(np.sum(self.output_activity)) + 1e-10)
        return char, confidence, top_idx

    def is_stable(self, required_consecutive=5):
        if len(self.output_window) < required_consecutive:
            return False
        recent = list(self.output_window)[-required_consecutive:]
        return len(set(recent)) == 1

    def get_forced_output(self):
        if not self.output_counter:
            return self.default_char
        best_idx = max(self.output_counter, key=self.output_counter.get)
        return self.reverse_char_map.get(best_idx, self.default_char)

    def apply_stdp_update(self, target_idx, reward_sign):
        lr_in = self._dtype(0.0005 if reward_sign > 0 else -0.0003)
        lr_out = self._dtype(0.0003 if reward_sign > 0 else -0.0002)

        for (out_idx, neuron_idx), trace_val in self._eligibility.items():
            if out_idx != target_idx:
                continue
            if neuron_idx < self.num_neurons:
                if neuron_idx < 256:
                    self.input_weights[neuron_idx, neuron_idx % 256] += lr_in * self._dtype(trace_val)
                self.output_weights[out_idx, neuron_idx] += lr_out * self._dtype(trace_val)

        self.input_weights = np.clip(self.input_weights, -1.0, 1.0)
        self.output_weights = np.clip(self.output_weights, -0.5, 0.5)
        self.recurrent_weights = np.clip(self.recurrent_weights, -0.3, 0.3)
        self._prune_eligibility()

    def get_activity_summary(self):
        active_indices = np.where(self.output_activity > 0.01)[0]
        result = {}
        for idx in active_indices[:20]:
            char = self.reverse_char_map.get(int(idx), '?')
            result[char] = float(self.output_activity[int(idx)])
        return result

    def get_memory_estimate(self):
        w_mem = (self.input_weights.nbytes + self.recurrent_weights.nbytes
                 + self.output_weights.nbytes) / 1e6
        e_mem = len(self._eligibility) * 16 / 1e6
        return {'weights_mb': round(w_mem, 1), 'eligibility_mb': round(e_mem, 3),
                'eligibility_entries': len(self._eligibility)}