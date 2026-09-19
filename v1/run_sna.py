#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SNA v4 - Simulated Neuron Architecture
Semantic Pointer Architecture + SNN Decoder + Predictive Coding + WTA Convergence
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

CONFIG_TEMPLATE = """sna:
  num_neurons: 10000
  avg_degree: 200
  time_step_ms: 1
  neuron_model: izhikevich
  distribution:
    regular: 0.6
    bursting: 0.3
    fast: 0.1
  dendrite_compartments: 2

stdp:
  a_plus: 0.01
  a_minus: 0.012
  tau_plus_ms: 10
  tau_minus_ms: 10
  dopamine_k: 1.0
  history_window_ms: 100

stp:
  u_base: 0.5
  tau_facil_ms: 100
  tau_rec_ms: 800

homeostasis:
  target_firing_rate_hz: 2.0
  window_ms: 1000
  plasticity_strength: 0.01

eligibility:
  lambda: 0.9
  eta: 0.01

working_memory:
  num_slots: 7
  decay_ms: 500

attention:
  num_attention_neurons: 100

world_model:
  hidden_neurons: 500
  prediction_window_ms: 10
  pred_coding_levels: 3
  pred_hidden_size: 300

semantic_pointer:
  spa_dim: 512
  neurons_per_concept: 12
  composition_depth: 3
  use_composition: true
  context_window: 6

wta_convergence:
  wta_decay: 0.95
  wta_inhibition: 0.15
  wta_leader_ratio: 1.5
  wta_stable_time: 50
  wta_abs_threshold: 0.8
  convergence_min_wait: 80
  convergence_max_wait: 5000

memory:
  instant_capacity: 10000000
  stm_capacity: 7
  consolidation_threshold: 0.6
  pattern_dim: 128

decoder:
  decoder_neurons: 1000
  char_vocab_size: 12000
  output_stability_consecutive: 5
  output_silence_steps: 200
  max_output_chars: 500

oscillatory:
  gamma_freq: 40.0
  theta_freq: 5.0
  enable_binding: true

self_model:
  dim: 128

forgetting:
  rate_per_10000steps: 0.95
  threshold: 0.01
  scan_interval_steps: 10000

sleep:
  interval_hours: 8
  duration_hours: 1
  replay_speedup: 5
  pruning_ratio: 0.05

output:
  streaming_window_ms: 50
  similarity_threshold: 0.7
  consecutive_windows: 3

embodiment:
  virtual_env: true
  terminal: true
  web_search: true
  env_config:
    world_size: 10x10
    objects:
      - tree
      - house
      - rock
"""


def _create_config(neuron_count=10000, force=False):
    config_path = os.path.join(SCRIPT_DIR, "config.temp.yaml")
    content = CONFIG_TEMPLATE.replace('num_neurons: 10000', f'num_neurons: {neuron_count}')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return config_path


def _run_sna(args):
    config_path = args.config
    if not config_path or not os.path.exists(config_path):
        config_path = _create_config(neuron_count=args.neurons)

    from python.main import SNABrainV6
    from python.utils.config_loader import load_config

    config = load_config(config_path)
    if args.neurons != 10000:
        config['sna']['num_neurons'] = args.neurons

    brain = SNABrainV6(config)
    brain.run()


def main():
    parser = argparse.ArgumentParser(
        description='SNA v4 - Semantic Pointers + WTA + SNN Decoder + Predictive Coding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sna.py                    # 10000 neurons (default)
  python run_sna.py -n 5000            # 5000 neurons
  python run_sna.py -n 1000            # 1000 neurons (quick test)
  python run_sna.py -n 50000           # 50000 neurons
  python run_sna.py -c my_config.yaml  # custom config
        """
    )

    parser.add_argument('-n', '--neurons', type=int, default=10000,
                       help='neuron count (default: 10000)')
    parser.add_argument('-c', '--config', type=str, default=None,
                       help='config file path')

    args = parser.parse_args()

    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [SYS ] {'=' * 50}")
    print(f"[{ts}] [SYS ]   SNA v4 - Simulated Neuron Architecture")
    print(f"[{ts}] [SYS ]   Neurons: {args.neurons}")
    print(f"[{ts}] [SYS ]   SPA: 512-dim vectors, 12 neurons/concept")
    print(f"[{ts}] [SYS ]   WTA Convergence with leader-ratio detection")
    print(f"[{ts}] [SYS ]   1000-neuron SNN Decoder for output")
    print(f"[{ts}] [SYS ]   3-level Predictive Coding hierarchy")
    print(f"[{ts}] [SYS ]   Self-model with internal state tracking")
    print(f"[{ts}] [SYS ]   Gamma oscillatory binding (40Hz)")
    print(f"[{ts}] [SYS ] {'=' * 50}")

    _run_sna(args)


if __name__ == "__main__":
    main()