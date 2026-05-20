import numpy as np
from collections import deque
from .snn_decoder import SNNDecoder


class OutputLayer:
    def __init__(self, population, config):
        self.pop = population
        self.total_neurons = population.size()

        self.decoder = SNNDecoder({
            'decoder_neurons': config.get('decoder_neurons', 1000),
            'char_vocab_size': config.get('char_vocab_size', 12000),
        })

        self.stability_required = config.get('output_stability_consecutive', 5)
        self.silence_threshold = config.get('output_silence_steps', 200)
        self.max_output_chars = config.get('max_output_chars', 500)

        self.output_buffer = []
        self.output_chars = 0
        self.silence_counter = 0
        self.clear_signal = False

        self.recent_activity = deque(maxlen=50)
        self.output_history = deque(maxlen=200)
        self.reward_history = deque(maxlen=20)

    def encode_conclusion(self, conclusion_data):
        neurons = conclusion_data.get('neurons', [])
        vector = np.zeros(256)
        for i, nid in enumerate(neurons[:128]):
            idx = nid % 256
            vector[idx] += (1.0 / (i + 1))

        activity = conclusion_data.get('activity', {})
        for nid, val in activity.items():
            idx = int(nid) % 256
            vector[idx] += float(val)

        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector /= norm

        return vector

    def stream_from_conclusion(self, conclusion_data, fires, current_step):
        vector = self.encode_conclusion(conclusion_data)
        char, confidence, top_idx = self.decoder.forward(vector, fires)

        self.recent_activity.append({
            'step': current_step,
            'char': char,
            'confidence': confidence,
            'top_idx': top_idx,
        })

        if self.decoder.is_stable(self.stability_required):
            self.output_buffer.append(char)
            self.output_chars += 1
            self.silence_counter = 0
            if self.reward_history:
                last_reward = self.reward_history[-1]
                self.decoder.apply_stdp_update(
                    top_idx, 1.0 if last_reward > 0 else -0.3)

        elif char == self.output_buffer[-1] if self.output_buffer else False:
            pass
        else:
            self.silence_counter += 1

        self._check_limits()
        return self._get_stream_output()

    def _check_limits(self):
        if self.silence_counter > self.silence_threshold:
            self.clear_signal = True
        if self.output_chars >= self.max_output_chars:
            self.clear_signal = True

    def _get_stream_output(self):
        new_chars = ''.join(self.output_buffer[-1:] if self.output_buffer else [])
        return {
            'new_char': new_chars,
            'full_text': ''.join(self.output_buffer),
            'is_done': self.clear_signal,
            'char_count': self.output_chars,
            'forced_output': self.decoder.get_forced_output() if self.clear_signal else None,
        }

    def force_output(self):
        forced = self.decoder.get_forced_output()
        self.output_buffer.append(forced)
        return ''.join(self.output_buffer)

    def record_reward(self, reward_sign):
        self.reward_history.append(reward_sign)

    def reset_for_new_round(self):
        self.output_buffer = []
        self.output_chars = 0
        self.silence_counter = 0
        self.clear_signal = False
        self.recent_activity.clear()
        self.decoder.reset_state()

    def get_decoder_activity(self):
        return self.decoder.get_activity_summary()

    def get_output_stats(self):
        return {
            'chars_produced': self.output_chars,
            'buffer_size': len(self.output_buffer),
            'silence_count': self.silence_counter,
            'clear_signal': self.clear_signal,
            'current_text': ''.join(self.output_buffer),
        }