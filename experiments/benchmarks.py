import numpy as np
from typing import Dict, List, Tuple
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentReport


class NavigationBenchmark:
    def __init__(self, world_sizes=[5.0, 10.0, 15.0], n_seeds_per=5):
        self.world_sizes = world_sizes
        self.n_seeds_per = n_seeds_per

    def run(self) -> Dict[str, ExperimentReport]:
        results = {}
        for ws in self.world_sizes:
            config = ExperimentConfig(
                n_neurons=32000,
                n_episodes=200,
                n_steps_per_episode=50,
                n_seeds=self.n_seeds_per,
                world_size=ws,
                seed_offset=42,
            )
            runner = ExperimentRunner(config)
            report = runner.run_all()
            results[f'world_size_{ws}'] = report
        return results

    def compare(self, results: Dict[str, ExperimentReport]) -> Dict:
        comparison = {}
        for label, report in results.items():
            agg = report.aggregate
            comparison[label] = {
                'success_rate': agg.get('success_rate_mean', 0),
                'phi_mean': agg.get('mean_phi_mean', 0),
                'phi_std': agg.get('mean_phi_std', 0),
                'consciousness_mean': agg.get('mean_consciousness_mean', 0),
                'reproducibility': report.phi_stability.get('reproducibility_score', 0),
                'phi_behavior_corr': report.correlation_analysis.get('pearson_r', 0),
            }
        return comparison


class LanguageBenchmark:
    def __init__(self, test_sentences: List[str] = None):
        self.test_sentences = test_sentences or [
            "the food is near the wall",
            "move left avoid danger",
            "see red object feel good",
            "take reward think fast",
            "I know you see me",
        ]

    def run_single_input(self, sentence: str, brain, world, n_steps=300) -> Dict:
        import core_cpp
        from python.cognitive.unified_consciousness import compute_consciousness_from_cortical_brain

        brain.reset_workspace()
        brain.inject_text(sentence)
        brain.inject_reward(0.5)

        phi_trace = []
        ignition_trace = []
        outputs = []

        for step in range(n_steps):
            sp = world.get_sensory_package()
            brain.inject_multi_modal(sp["visual"], sp["auditory"],
                                      sp["tactile"], sp["vestibular"],
                                      sp["place_cells"])
            brain.step(0.0)

            cs = brain.read_consciousness()
            phi_trace.append(cs.phi)
            ignition_trace.append(cs.global_ignition)

            if step % 20 == 0:
                outputs.append(brain.read_output_text())

        consciousness, components = compute_consciousness_from_cortical_brain(brain)

        return {
            'input': sentence,
            'phi_mean': float(np.mean(phi_trace[-50:])),
            'phi_std': float(np.std(phi_trace[-50:])),
            'ignition_mean': float(np.mean(ignition_trace[-50:])),
            'consciousness': consciousness,
            'unique_outputs': len(set(outputs)),
            'outputs': outputs[-3:],
        }

    def run(self, brain, world, n_inputs=5) -> List[Dict]:
        results = []
        for sentence in self.test_sentences[:n_inputs]:
            result = self.run_single_input(sentence, brain, world)
            results.append(result)
        return results


class ConsciousnessStabilityBenchmark:
    def __init__(self, n_runs=10, steps_per_run=500, perturbation_levels=[0.0, 0.1, 0.5]):
        self.n_runs = n_runs
        self.steps_per_run = steps_per_run
        self.perturbation_levels = perturbation_levels

    def run(self) -> Dict:
        results = {}
        for noise in self.perturbation_levels:
            config = ExperimentConfig(
                n_neurons=32000,
                n_episodes=1,
                n_steps_per_episode=self.steps_per_run,
                n_seeds=self.n_runs,
                seed_offset=100,
            )
            runner = ExperimentRunner(config)
            report = runner.run_all()

            agg = report.aggregate
            results[f'noise_{noise}'] = {
                'phi_mean': agg.get('mean_phi_mean', 0),
                'phi_std_cross_seed': agg.get('mean_phi_std', 0),
                'phi_cov': report.phi_stability.get('phi_cov', 0),
                'reproducibility': report.phi_stability.get('reproducibility_score', 0),
            }
        return results


def run_all_benchmarks() -> Dict:
    print("=" * 60)
    print("SNA BENCHMARK SUITE")
    print("=" * 60)

    results = {}

    print("\n[1/3] Navigation Benchmark...")
    nav = NavigationBenchmark(world_sizes=[10.0], n_seeds_per=3)
    nav_results = nav.run()
    nav_comparison = nav.compare(nav_results)
    results['navigation'] = nav_comparison
    print(f"  World 10.0: success={nav_comparison['world_size_10.0']['success_rate']:.4f}")
    print(f"  World 10.0: phi={nav_comparison['world_size_10.0']['phi_mean']:.4f}")

    print("\n[2/3] Consciousness Stability Benchmark...")
    stab = ConsciousnessStabilityBenchmark(n_runs=3, steps_per_run=300)
    stab_results = stab.run()
    results['stability'] = stab_results
    for level, data in stab_results.items():
        print(f"  {level}: phi={data['phi_mean']:.4f}, reproducibility={data['reproducibility']:.4f}")

    print("\n[3/3] Cross-benchmark summary...")
    summary = {
        'navigation_phi_mean': nav_comparison['world_size_10.0']['phi_mean'],
        'navigation_success': nav_comparison['world_size_10.0']['success_rate'],
        'phi_stability_cov': stab_results['noise_0.0']['phi_cov'],
        'reproducibility': stab_results['noise_0.0']['reproducibility'],
        'phi_behavior_corr': nav_comparison['world_size_10.0']['phi_behavior_corr'],
    }
    results['summary'] = summary

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    return results


if __name__ == '__main__':
    run_all_benchmarks()