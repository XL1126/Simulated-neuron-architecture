import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.utils.config_loader import load_config
from python.utils.tensor_utils import to_fixed_dim, normalize, cosine_sim
from python.layers.input_layer import InputLayer
from python.layers.primary_layer import PrimaryLayer
from python.layers.core_layer import CoreLayer
from python.layers.memory_layer import MemoryLayer
from python.layers.output_layer import OutputLayer
from python.layers.snn_decoder import SNNDecoder
from python.cognitive.working_memory import WorkingMemory
from python.cognitive.attention import Attention
from python.cognitive.world_model import WorldModel
from python.cognitive.metacognition import MetaCognition
from python.cognitive.predictive_coding import PredictiveCoding
from python.cognitive.self_model import SelfModel
from python.cognitive.oscillatory_binding import OscillatoryBinding
from python.cognitive.semantic_pointer import SemanticPointer
from python.cognitive.causal_graph import CausalGraph
from python.cognitive.consciousness_metrics import ConsciousnessMetrics
from python.reward.intrinsic_motivation import IntrinsicMotivation
from python.reward.credit_assignment import CreditAssignment
from python.embodiment.embodiment_interface import EmbodimentInterface
from python.embodiment.virtual_world import VirtualWorld, VirtualWorldEpisodeRunner
from python.interaction.nonblocking_io import NonBlockingInput
from python.interaction.stream_output import StreamOutput
from python.utils.dashboard import Dashboard

try:
    import core_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class SNABrainV6:
    def __init__(self, config=None):
        self.config = config or load_config()

        if not CPP_AVAILABLE:
            raise RuntimeError('C++ core (core_cpp) required. Run cmake build first.')

        sna_cfg = self.config['sna']
        n_total = sna_cfg['num_neurons']
        self.pop = core_cpp.NeuronPopulation(n_total, sna_cfg['avg_degree'])
        self._init_stdp()

        dim = 512

        self.spa_mapper = core_cpp.SPANeuralMapper(n_total, dim, 12)
        self.gw = core_cpp.GlobalWorkspacePopulation(200, 10)

        self.spa = SemanticPointer(dim=dim)
        self.input_layer = InputLayer(self.pop,
            self.config.get('semantic_pointer', {}))
        self.primary_layer = PrimaryLayer(self.pop)
        self.core_layer = CoreLayer(self.pop,
            self.config.get('wta_convergence', {}))
        self.memory_layer = MemoryLayer(self.pop,
            self.config.get('memory', {}))
        self.snn_decoder = SNNDecoder(self.config.get('decoder', {}))
        self.output_layer = OutputLayer(self.pop,
            self.config.get('decoder', {}))

        self.working_memory = WorkingMemory(self.pop,
            self.config.get('working_memory', {}).get('num_slots', 7))
        self.attention = Attention(self.pop,
            self.config.get('attention', {}).get('num_attention_neurons', 100))
        self.world_model = WorldModel(self.pop, 500, 10)
        self.metacognition = MetaCognition(self.pop)
        self.predictive_coding = PredictiveCoding(self.pop, num_levels=3, hidden_size=300)
        self.self_model = SelfModel(self.pop, dim=256)
        self.oscillatory = OscillatoryBinding(n_total,
            gamma_freq=self.config.get('oscillatory', {}).get('gamma_freq', 40.0))
        self.causal_graph = CausalGraph(self.spa, dim=256)
        self.consciousness = ConsciousnessMetrics()

        self.intrinsic_motivation = IntrinsicMotivation()
        self.credit_assignment = CreditAssignment(
            lambda_decay=self.config.get('eligibility', {}).get('lambda', 0.9),
            eta=self.config.get('eligibility', {}).get('eta', 0.01))

        self.virtual_world = VirtualWorld()
        self.vw_runner = VirtualWorldEpisodeRunner(
            self.virtual_world, self.pop, self.spa_mapper)
        self.embodiment = EmbodimentInterface(self.config.get('embodiment', {}))
        self.stream_output = StreamOutput()

        self.dopamine = 0.5
        self.step_counter = 0
        self.current_input_text = ''
        self.waiting_for_input = False
        self.total_reward = 0.0
        self.vw_step_counter = 0

        self.sleep_every = 1000
        self.last_sleep_step = 0
        self.metrics_print_every = 100

        self._register_core_concepts()
        self.core_layer.build_topology()

    def _init_stdp(self):
        cfg = core_cpp.STDPConfig()
        stdp = self.config['stdp']
        cfg.a_plus = stdp['a_plus']
        cfg.a_minus = stdp['a_minus']
        cfg.tau_plus_ms = stdp['tau_plus_ms']
        cfg.tau_minus_ms = stdp['tau_minus_ms']
        cfg.dopamine_k = stdp['dopamine_k']
        cfg.history_window_ms = stdp['history_window_ms']
        self.pop.set_stdp_config(cfg)

    def _register_core_concepts(self):
        core = ['我', '你', '是', '不', '有', '在', '和', '的', '猫', '狗',
                '红色', '蓝色', '绿色', '方块', '圆形', '左', '右', '上', '下',
                '高兴', '难过', '喜欢', 'move_up', 'move_down', 'move_left',
                'move_right', 'grab', 'wait']
        for c in core:
            self.spa_mapper.add_concept(c)
            self.spa.add_symbol(c, seed=hash(c) % (2**31))
            self.causal_graph.add_node(c)

    def run(self):
        io_handler = NonBlockingInput()
        io_handler.start()
        dashboard = Dashboard(self)
        dashboard.start()
        print('[SNA v6] ===== CONSCIOUS ARCHITECTURE ====')
        print(f'[SNA v6] Neurons: {self.pop.size()} | SPA dim: 512 | GW: 200')
        print('[SNA v6] C++ SPANeuralMapper + GlobalWorkspace')
        print('[SNA v6] CausalGraph + Recursive SelfModel')
        print('[SNA v6] VirtualWorld: 8x8 grid, color/shape tasks')
        print('[SNA v6] ================================\n')
        print('[SNA v6] Type text, +/- for feedback, ctrl+c to quit\n')

        while True:
            self.step_counter += 1

            if self.waiting_for_input:
                text = io_handler.read()
                if text:
                    s = text.strip()
                    if s == '+':
                        self.total_reward += 1.0
                        self.output_layer.record_reward(1.0)
                    elif s == '-':
                        self.total_reward -= 0.5
                        self.output_layer.record_reward(-1.0)
                    else:
                        self._process_input(s)
                    self.waiting_for_input = False
                time.sleep(0.01)
                continue

            self._step_network()
            self._gw_cycle()
            self._virtual_world_tick()

            if self.core_layer.check_convergence():
                self._output_phase()

            self._sleep_cycle()
            self._print_metrics()

    def _process_input(self, text):
        self.current_input_text = text
        self.core_layer.reset_for_next_round()
        self.output_layer.reset_for_new_round()
        self.snn_decoder.reset_state()
        self.gw.reset()

        events = self.input_layer.encode_text(text)
        event_spikes = []
        concepts = set()
        for ev in events:
            event_spikes.append({'target_id': ev.get('neuron_id', 0),
                                  'strength': ev.get('strength', 1.0),
                                  'time_ms': 0, 'source': 'input'})
            w = ev.get('word', ev.get('char', ''))
            if w:
                concepts.add(w)
                self.spa_mapper.add_concept(w)
                self.causal_graph.add_node(w)

        self.primary_layer.inject_spikes(event_spikes)

        vec = self.input_layer.get_context_vector()
        vec = to_fixed_dim(vec, 512)

        mapper_spikes = self.spa_mapper.map_vector_to_spikes(vec.tolist())
        for nid, strength in mapper_spikes:
            self.pop.inject_spike(nid % self.pop.size(), strength, 1)

        print(f'\n[SNA v6] Input: {text[:80]}')
        print(f'[SNA v6]   concepts: {", ".join(sorted(list(concepts))[:8])}')

    def _step_network(self):
        fires = self.pop.get_current_fires()
        self.core_layer.step(fires, self.step_counter)

        active_nids = {f.neuron_id for f in fires}
        self.attention.compute_gains()
        self.world_model.predict_and_compare()
        pc_fe = self.predictive_coding.step(fires)

        total_pred_err = self.world_model.get_prediction_error() * 0.4 + pc_fe * 0.6
        firing_rate = len(fires) / max(1, self.pop.size())

        self.self_model.update(self.step_counter, self.dopamine, firing_rate,
                               total_pred_err, self.current_input_text)
        self.self_model.predict_self()

        self.oscillatory.step(dt=1.0, active_neurons=active_nids)

        intrinsic_r = self.intrinsic_motivation.compute(
            total_pred_err, novelty=pc_fe,
            active_pattern=fires[:20] if fires else None)

        self.dopamine += (intrinsic_r - 0.3) * 0.1
        self.dopamine = max(0.0, min(1.0, self.dopamine))

        if len(fires) > 3:
            self.pop.set_dopamine(self.dopamine)

        self.pop.update(self.step_counter, 0.005, self.dopamine)
        self.credit_assignment.update_traces(self.pop)
        self.credit_assignment.apply_credit(self.pop, intrinsic_r - 0.5)
        self.memory_layer.step(self.step_counter)

        if self.step_counter % 100 == 0:
            self.primary_layer.form_bridge_synapses(threshold=3)
            self.spa_mapper.decay_eligibility(0.99)

    def _gw_cycle(self):
        fires = self.pop.get_current_fires()
        gw_inputs = [(f.neuron_id % self.gw.size(),
                      f.strength) for f in fires[:50]]
        winner = self.gw.step(gw_inputs, self.dopamine)

        phi = self.gw.get_phi_estimate()
        self.consciousness.update_phi(phi)

        if self.gw.is_ignited():
            broadcasts = self.gw.broadcast(self.pop.size())
            for b in broadcasts:
                self.pop.inject_spike(b.target_neuron_id, b.strength, b.delay_ms)

        winner_concept = self.spa_mapper.resolve_neuron(
            winner) if winner >= 0 else ''

    def _virtual_world_tick(self):
        self.vw_step_counter += 1
        if self.vw_step_counter < 100:
            return
        self.vw_step_counter = 0

        spike_events = self.virtual_world.to_spike_events()
        for ev in spike_events:
            self.pop.inject_spike(ev['target_id'] % self.pop.size(),
                                  ev.get('strength', 1.0), 1)

        fires = self.pop.get_current_fires()
        active_neurons = [f.neuron_id for f in fires[:10]]
        action_idx = (sum(active_neurons) % 7) if active_neurons else 0
        result = self.virtual_world.step(action_idx)

        if result['reward'] > 0:
            self.dopamine = min(1.0, self.dopamine + 0.3)
            self.total_reward += result['reward']

        self.self_model.is_self_generated(active_neurons, source_type='internal')

        feats = result['features']
        for task in self.virtual_world.tasks:
            if not task['done']:
                self.spa_mapper.add_concept(task['target_color'])
                self.spa_mapper.add_concept(task['target_shape'])

    def _output_phase(self):
        conclusion = self.core_layer.get_conclusion()
        output_chars = []

        for _ in range(200):
            fires = self.pop.get_current_fires()
            active_spikes = [(f.neuron_id, f.strength) for f in fires[:20]]

            broadcasts = self.gw.broadcast(self.pop.size())
            for b in broadcasts:
                self.pop.inject_spike(b.target_neuron_id, b.strength, b.delay_ms)

            if conclusion.get('leader', -1) >= 0:
                self.pop.add_synaptic_current(
                    conclusion['leader'] % self.pop.size(), 0.15)

            if active_spikes:
                reconstructed = self.spa_mapper.map_spikes_to_vector(active_spikes)
            else:
                reconstructed = np.zeros(512, dtype=np.float32)

            char, conf, idx = self.snn_decoder.forward(reconstructed)
            if self.snn_decoder.is_stable(5):
                output_chars.append(char)
                self.stream_output.emit(char)
                self.self_model.is_self_generated(
                    [f.neuron_id for f in fires[:10]], source_type='internal')

            if len(output_chars) > 100:
                break

            self.pop.update(self.step_counter, 0.003, self.dopamine)
            self.core_layer.wta.candidates *= 0.98

        final = ''.join(output_chars) if output_chars else '...'
        print(f'\n[SNA v6] Output: {final}')

        self.consciousness.compute_consciousness_level(
            self.gw.get_activity(), self.self_model)
        report = self.consciousness.get_consciousness_report()
        print(f'[SNA v6]   Phi: {report["phi"]:.4f} | '
              f'Consciousness: {report["consciousness_level"]:.3f} | '
              f'Self-err: {report["self_prediction_error"]:.4f}')

        thought = self.core_layer.get_thought_vector()
        self.memory_layer.save_replay(thought)

        if self.current_input_text:
            self.causal_graph.learn_from_transition(
                [self.current_input_text[:20]], [final[:20]], self.total_reward)

        self.gw.reset()
        self.waiting_for_input = True
        self.stream_output.flush()
        print()

    def _sleep_cycle(self):
        if self.step_counter - self.last_sleep_step < self.sleep_every:
            return

        print('\n[SNA v6] === SLEEP REPLAY ===')
        for _ in range(10):
            replay_pattern = self.memory_layer.get_replay()
            if replay_pattern is not None:
                nids = []
                for i, v in enumerate(replay_pattern.flatten()[:20]):
                    nid = int(abs(v) * 100) % self.pop.size()
                    nids.append(nid)
                self.pop.inject_spike_group(nids, 0.05, 1)

        self.pop.apply_sleep_cycle(self.step_counter)
        self.memory_layer.consolidate(f'sleep_{self.step_counter}',
            self.core_layer.get_thought_vector(), 0.6)
        self.last_sleep_step = self.step_counter
        print('[SNA v6] === SLEEP DONE ===\n')

    def _print_metrics(self):
        if self.step_counter % self.metrics_print_every != 0:
            return
        ws = self.self_model.get_self_description()
        report = self.consciousness.get_consciousness_report()
        print(f'\r[SNA v6] t={self.step_counter}ms | '
              f'Phi={report["phi"]:.4f} | '
              f'level={report["consciousness_level"]:.3f} | '
              f'self-err={ws["certainty"]:.3f} | '
              f'VW: {self.virtual_world.get_status()}',
              end='', flush=True)

    def get_status(self):
        fires = self.pop.get_current_fires()
        conclusion = self.core_layer.get_conclusion()
        output_stats = self.output_layer.get_output_stats()
        memory_stats = self.memory_layer.memory_stats()
        decoder = self.snn_decoder.get_activity_summary()
        self_desc = self.self_model.get_self_description()
        report = self.consciousness.get_consciousness_report()
        cg = self.causal_graph.get_summary()
        gwe = self.snn_decoder.get_memory_estimate()

        return {
            'step': self.step_counter,
            'neurons': self.pop.size(),
            'dopamine': round(self.dopamine, 3),
            'firing': len(fires),
            'wta_leader': conclusion.get('leader', -1),
            'converged': self.core_layer.check_convergence(),
            'output': output_stats,
            'memory': memory_stats,
            'decoder': decoder,
            'self': self_desc,
            'consciousness': report,
            'causal_graph': cg,
            'virtual_world': {
                'task_completed': self.virtual_world.task_completed,
                'current_task': self.virtual_world.current_task,
                'total_reward': self.virtual_world.total_reward,
            },
            'decoder_memory_mb': gwe,
        }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)
    brain = SNABrainV6(config)
    brain.run()


if __name__ == '__main__':
    main()