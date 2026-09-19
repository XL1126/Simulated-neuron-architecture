"""
SNA 统一运行入口 — 整合 CorticalBrain 和 SNABrainV6 的实验运行器。

用法:
    python run_experiment.py cortical     # 运行 CorticalBrain 标准实验
    python run_experiment.py sna          # 运行 SNABrainV6 实验 (需 binding 完整)
    python run_experiment.py benchmark    # 运行完整基准测试套件
    python run_experiment.py calibrate    # Phi-Behavior 校准
    python run_experiment.py quick        # 快速单 seed 验证运行
"""
import sys
import os
import time
import json
import argparse

os.environ["OMP_NUM_THREADS"] = "4"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'python'))
sys.path.insert(0, HERE)

import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def run_cortical_experiment(args):
    from experiments.experiment_runner import ExperimentRunner, ExperimentConfig

    config = ExperimentConfig(
        n_neurons=getattr(args, 'neurons', 32000),
        n_episodes=getattr(args, 'episodes', 200),
        n_steps_per_episode=getattr(args, 'steps_per_ep', 50),
        n_seeds=getattr(args, 'seeds', 5),
        seed_offset=getattr(args, 'seed_offset', 42),
        world_size=getattr(args, 'world_size', 10.0),
        omp_num_threads=getattr(args, 'omp', 4),
    )

    print("=" * 60)
    print("SNA CorticalBrain Experiment")
    print(f"  Neurons: {config.n_neurons:,}")
    print(f"  Episodes: {config.n_episodes} x {config.n_steps_per_episode} steps")
    print(f"  Seeds: {config.n_seeds} (offset={config.seed_offset})")
    print(f"  World: {config.world_size}x{config.world_size}")
    print("=" * 60)

    runner = ExperimentRunner(config)

    def progress(i, total, seed):
        print(f"  [{i+1}/{total}] Running seed={seed}...", flush=True)

    t_start = time.perf_counter()
    report = runner.run_all(progress_callback=progress)
    elapsed = time.perf_counter() - t_start

    report.print_summary()

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Avg per seed: {elapsed/config.n_seeds:.1f}s")

    filepath = os.path.join(config.output_dir, f'cortical_experiment_{time.strftime("%Y%m%d_%H%M%S")}.json')
    report.save(filepath)
    print(f"\n  Report saved: {filepath}")

    from python.cognitive.phi_validator import PhiCorrectionEngine

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Avg per seed: {elapsed/config.n_seeds:.1f}s")

    filepath = os.path.join(config.output_dir, f'cortical_experiment_{time.strftime("%Y%m%d_%H%M%S")}.json')
    report.save(filepath)
    print(f"\n  Report saved: {filepath}")

    return report


def run_sna_experiment(args):
    import core_cpp

    try:
        from python.main import SNABrainV6
    except Exception as e:
        print(f"ERROR: SNABrainV6 import failed: {e}")
        print("This likely means NeuronPopulation binding is incomplete.")
        print("Run 'python check_binding.py' to diagnose.")
        return None

    print("=" * 60)
    print("SNABrainV6 Experiment (Python + C++ mixed)")
    print("=" * 60)

    try:
        brain = SNABrainV6()
        print(f"  Neurons: {brain.pop.size()}")
        print(f"  SPA dim: 512 | GW: {brain.gw.size()}")
        print(f"  SNABrainV6 initialized successfully!")
        print(f"  (Full experiment not yet automated - use interactive mode)")
        return brain
    except Exception as e:
        print(f"  SNABrainV6 failed: {e}")
        return None


def run_benchmark(args):
    from experiments.benchmarks import run_all_benchmarks
    results = run_all_benchmarks()

    output_dir = os.path.join(HERE, 'experiment_results')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'benchmark_{time.strftime("%Y%m%d_%H%M%S")}.json')

    serializable = {}
    for k, v in results.items():
        if hasattr(v, 'to_dict'):
            serializable[k] = v.to_dict()
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Benchmark results saved: {filepath}")


def run_calibrate(args):
    from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
    from python.cognitive.phi_validator import (
        PhiCorrectionEngine,
        PhiBehaviorCalibration,
    )

    print("=" * 60)
    print("Phi-Behavior Calibration v4")
    print("=" * 60)

    config = ExperimentConfig(
        n_neurons=getattr(args, 'neurons', 32000),
        n_episodes=getattr(args, 'episodes', 300),
        n_steps_per_episode=getattr(args, 'steps_per_ep', 50),
        n_seeds=10,
        seed_offset=42,
        sensory_injection_interval=3,
    )

    runner = ExperimentRunner(config)

    def progress(i, total, seed):
        print(f"  [{i+1}/{total}] Running seed={seed}...")

    t_start = time.perf_counter()
    report = runner.run_all(progress_callback=progress)
    elapsed = time.perf_counter() - t_start

    print(f"\n  Completed {config.n_seeds} seeds in {elapsed:.1f}s")

    report.print_summary()

    output_dir = os.path.join(HERE, 'experiment_results')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'calibration_{time.strftime("%Y%m%d_%H%M%S")}.json')
    report.save(filepath)
    print(f"\n  Calibration saved: {filepath}")


def run_quick(args):
    import core_cpp

    print("=" * 60)
    print("SNA Quick Verification (1 seed, 100 episodes)")
    print("=" * 60)

    concepts = [
        "self", "world", "move", "see", "eat",
        "good", "bad", "near", "far", "red",
        "blue", "green", "yellow", "up", "down",
        "left", "right", "object", "food", "wall",
        "empty", "reward", "danger", "safe", "want",
        "think", "feel", "know", "I", "you",
    ]

    brain = core_cpp.CorticalBrain(32000, concepts)
    brain.set_seed(42)
    print(f"  Neurons: {brain.total_neurons():,}")
    regions = brain.get_regions()
    for reg in regions[:5]:
        print(f"  {reg.name}: {reg.n_neurons} neurons")
    print(f"  ... ({len(regions)} regions total)")

    from experiments.experiment_runner import EnhancedVirtualWorld
    world = EnhancedVirtualWorld(10.0, seed=42)

    sp = world.get_sensory_package()
    brain.inject_multi_modal(sp["visual"], sp["auditory"],
                              sp["tactile"], sp["vestibular"],
                              sp["place_cells"])
    brain.inject_text(world.describe())

    phi_history = []
    ignition_history = []
    successes = 0
    steps = 100 * 100

    for step in range(steps):
        reward = 0.0
        if step % 10 == 0:
            sp = world.get_sensory_package()
            brain.inject_multi_modal(sp["visual"], sp["auditory"],
                                      sp["tactile"], sp["vestibular"],
                                      sp["place_cells"])
            brain.inject_text(world.describe())
            world.tick()

        motor = brain.read_motor_output()
        reward = world.try_interact(motor)
        if reward > 0:
            successes += 1
        brain.step(reward)

        cs = brain.read_consciousness()
        phi_history.append(cs.phi)
        ignition_history.append(cs.global_ignition)

    from python.cognitive.unified_consciousness import compute_consciousness_from_cortical_brain
    consciousness, components = compute_consciousness_from_cortical_brain(brain)

    print(f"\n  ── Results (100 episodes) ──")
    print(f"  Phi mean:          {np.mean(phi_history[-100:]):.4f}")
    print(f"  Phi std:           {np.std(phi_history[-100:]):.4f}")
    print(f"  Ignition mean:     {np.mean(ignition_history[-100:]):.4f}")
    print(f"  Consciousness:     {consciousness:.4f}")
    print(f"  Components:")
    print(f"    IIT:             {components['iit_component']:.4f}")
    print(f"    GWT:             {components['gwt_component']:.4f}")
    print(f"    Predictive:      {components['predictive_component']:.4f}")
    print(f"    FP Salience:     {components['fp_component']:.4f}")
    print(f"    Vividness:       {components['vividness_component']:.4f}")
    print(f"  Successes:         {successes} / {steps} ({successes/max(1,steps)*100:.1f}%)")
    print(f"  Vividness:         {brain.get_perceptual_vividness():.4f}")
    print(f"  FP Salience:       {brain.get_first_person_salience():.4f}")
    print(f"  Output:            {brain.read_output_text()}")
    print(f"  Self narrative:    {brain.get_self_narrative()}")


def main():
    parser = argparse.ArgumentParser(
        description='SNA Unified Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py quick        Fast single-seed verification
  python run_experiment.py cortical     Standard CorticalBrain experiment
  python run_experiment.py benchmark    Full benchmark suite
  python run_experiment.py calibrate    Phi-Behavior calibration (10 seeds)
  python run_experiment.py sna          SNABrainV6 experiment (beta)
        """)

    parser.add_argument('mode', nargs='?', default='quick',
                        choices=['quick', 'cortical', 'sna', 'benchmark', 'calibrate'],
                        help='运行模式')

    parser.add_argument('--neurons', type=int, default=32000,
                        help='神经元数量 (default: 32000)')
    parser.add_argument('--episodes', type=int, default=200,
                        help='训练 episode 数 (default: 200)')
    parser.add_argument('--steps-per-ep', type=int, default=50,
                        help='每个 episode 步数 (default: 50)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='seed 数量 (default: 5)')
    parser.add_argument('--seed-offset', type=int, default=42,
                        help='seed 起始偏移 (default: 42)')
    parser.add_argument('--world-size', type=float, default=10.0,
                        help='虚拟世界大小 (default: 10.0)')
    parser.add_argument('--omp', type=int, default=4,
                        help='OpenMP 线程数 (default: 4)')

    args = parser.parse_args()

    if args.mode == 'quick':
        run_quick(args)
    elif args.mode == 'cortical':
        run_cortical_experiment(args)
    elif args.mode == 'sna':
        run_sna_experiment(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)
    elif args.mode == 'calibrate':
        run_calibrate(args)


if __name__ == '__main__':
    main()