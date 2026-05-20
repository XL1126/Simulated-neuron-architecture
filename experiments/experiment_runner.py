import sys
import os
import time
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import core_cpp
from python.cognitive.unified_consciousness import (
    compute_consciousness_from_cortical_brain,
    ConsciousnessCalibrator,
    cross_validate_consciousness,
)
from python.cognitive.phi_validator import (
    PhiCorrectionEngine,
    PhiBehaviorCalibration,
    PhiValidator,
    check_self_referential_loop,
    check_ignition_health,
)


@dataclass
class ExperimentConfig:
    n_neurons: int = 32000
    n_episodes: int = 600
    n_steps_per_episode: int = 100
    n_seeds: int = 10
    seed_offset: int = 0
    world_size: float = 10.0
    omp_num_threads: int = 4
    enable_phase_d: bool = True
    output_dir: str = ""
    enable_correction_engine: bool = True
    correction_window_size: int = 300
    sensory_injection_interval: int = 3
    calibration_start_step: int = 200
    calibration_interval: int = 100

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'experiment_results')
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def total_steps(self) -> int:
        return self.n_episodes * self.n_steps_per_episode


@dataclass
class SeedResult:
    seed: int
    phi_history: List[float] = field(default_factory=list)
    phi_corrected_history: List[float] = field(default_factory=list)
    ignition_history: List[float] = field(default_factory=list)
    ignition_corrected_history: List[float] = field(default_factory=list)
    success_history: List[int] = field(default_factory=list)
    consciousness_levels: List[float] = field(default_factory=list)
    consciousness_corrected_levels: List[float] = field(default_factory=list)
    self_pred_errors: List[float] = field(default_factory=list)
    self_pred_errors_corrected: List[float] = field(default_factory=list)
    vividness_history: List[float] = field(default_factory=list)
    fp_salience_history: List[float] = field(default_factory=list)
    output_texts: List[str] = field(default_factory=list)
    goal_progress: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    total_steps: int = 0
    final_phi: float = 0.0
    final_phi_corrected: float = 0.0
    final_consciousness: float = 0.0
    final_consciousness_corrected: float = 0.0
    total_successes: int = 0
    loop_suspected_rate: float = 0.0
    ignition_saturated_rate: float = 0.0
    sensory_deprived_rate: float = 0.0
    correction_engine_stats: dict = field(default_factory=dict)
    calibrator_diagnostics: dict = field(default_factory=dict)

    def compute_summary(self) -> dict:
        if not self.phi_history:
            return {}
        corrected_phis = self.phi_corrected_history if self.phi_corrected_history else self.phi_history
        corrected_consc = self.consciousness_corrected_levels if self.consciousness_corrected_levels else self.consciousness_levels
        corrected_igns = self.ignition_corrected_history if self.ignition_corrected_history else self.ignition_history
        corrected_errs = self.self_pred_errors_corrected if self.self_pred_errors_corrected else self.self_pred_errors

        tail = min(100, len(self.phi_history))
        return {
            'mean_phi': float(np.mean(self.phi_history[-tail:])),
            'std_phi': float(np.std(self.phi_history[-tail:])),
            'mean_phi_corrected': float(np.mean(corrected_phis[-tail:])),
            'mean_ignition': float(np.mean(self.ignition_history[-tail:])),
            'mean_ignition_corrected': float(np.mean(corrected_igns[-tail:])),
            'mean_vividness': float(np.mean(self.vividness_history[-tail:])),
            'mean_fp_salience': float(np.mean(self.fp_salience_history[-tail:])),
            'mean_consciousness': float(np.mean(self.consciousness_levels[-20:])) if self.consciousness_levels else 0.0,
            'mean_consciousness_corrected': float(np.mean(corrected_consc[-20:])) if corrected_consc else 0.0,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, len(self.success_history)),
            'mean_self_pred_error': float(np.mean(self.self_pred_errors[-tail:])),
            'mean_self_pred_error_corrected': float(np.mean(corrected_errs[-tail:])),
            'elapsed_seconds': self.elapsed_seconds,
            'steps_per_second': self.total_steps / max(0.001, self.elapsed_seconds),
            'loop_suspected_rate': self.loop_suspected_rate,
            'ignition_saturated_rate': self.ignition_saturated_rate,
            'sensory_deprived_rate': self.sensory_deprived_rate,
            'correction_gain': self.correction_engine_stats.get('correction_gain', 0.0),
        }


@dataclass
class ExperimentReport:
    config: ExperimentConfig
    seed_results: List[SeedResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    phi_stability: dict = field(default_factory=dict)
    correlation_analysis: dict = field(default_factory=dict)
    diagnostic_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'config': asdict(self.config),
            'n_seeds_completed': len(self.seed_results),
            'aggregate': self.aggregate,
            'phi_stability': self.phi_stability,
            'correlation_analysis': self.correlation_analysis,
            'diagnostic_summary': self.diagnostic_summary,
            'per_seed': [r.compute_summary() for r in self.seed_results],
        }

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        print("\n" + "=" * 64)
        print("SNA EXPERIMENT REPORT v7")
        print("=" * 64)
        agg = self.aggregate
        diag = self.diagnostic_summary

        n = len(self.seed_results)
        print(f"\n  Seeds completed:       {n}/{self.config.n_seeds}")
        print(f"  Sensory Interval:      {self.config.sensory_injection_interval} steps")
        print(f"  Correction Engine:     {'ENABLED' if self.config.enable_correction_engine else 'DISABLED'}")

        print(f"\n  ---- Phi (Information Integration) ----")
        print(f"    Raw:       {agg.get('mean_phi_mean', 0):.4f} ± {agg.get('mean_phi_std', 0):.4f}")
        if agg.get('mean_phi_corrected_mean', 0) > 0:
            print(f"    Corrected: {agg.get('mean_phi_corrected_mean', 0):.4f} ± {agg.get('mean_phi_corrected_std', 0):.4f}")

        print(f"\n  ---- GWT Ignition ----")
        print(f"    Raw:       {agg.get('mean_ignition_mean', 0):.4f} ± {agg.get('mean_ignition_std', 0):.4f}")
        print(f"    Ign Var:   {diag.get('mean_ignition_variance', 0):.6f}")
        if agg.get('mean_ignition_corrected_mean', 0) > 0:
            print(f"    Corrected: {agg.get('mean_ignition_corrected_mean', 0):.4f}")

        print(f"\n  ---- Consciousness Level ----")
        print(f"    Raw Mean:  {agg.get('mean_consciousness_mean', 0):.4f}")
        if agg.get('mean_consciousness_corrected_mean', 0) > 0:
            print(f"    Corr Mean: {agg.get('mean_consciousness_corrected_mean', 0):.4f}")

        print(f"\n  ---- Task Performance ----")
        print(f"    Success Rate: {agg.get('success_rate_mean', 0):.4f}")
        if agg.get('mean_vividness_mean', 0) > 0:
            print(f"    Vividness:    {agg.get('mean_vividness_mean', 0):.4f}")

        print(f"\n  ---- Self-Prediction Error ----")
        print(f"    Raw:       {agg.get('mean_self_pred_error_mean', 0):.4f}")
        if agg.get('mean_self_pred_error_corrected_mean', 0) > 0:
            print(f"    Corrected: {agg.get('mean_self_pred_error_corrected_mean', 0):.4f}")

        print(f"\n  ---- Phi Stability (cross-seed) ----")
        stab = self.phi_stability
        print(f"    CoV (Phi Raw):     {stab.get('phi_cov', 0):.4f}")
        print(f"    CoV (Phi Corr):    {stab.get('phi_corrected_cov', 0):.4f}")
        print(f"    Reproducibility:   {stab.get('reproducibility_score', 0):.4f}")

        print(f"\n  ---- Phi-Behavior Correlation ----")
        corr = self.correlation_analysis
        print(f"    Pearson r (Raw):   {corr.get('pearson_r', 0):.4f}")
        print(f"    Pearson r (Corr):  {corr.get('pearson_r_corrected', 0):.4f}")
        print(f"    Spearman rho:      {corr.get('spearman_rho', 0):.4f}")
        print(f"    p-value:           {corr.get('p_value', 0):.4f}")
        print(f"    Interpretation:    {corr.get('interpretation', 'N/A')}")

        print(f"\n  ---- Cross-Validation (C++ ↔ Python) ----")
        xv = diag.get('cross_validation', {})
        print(f"    Aligned:           {xv.get('aligned_ratio', 0):.3f}")
        print(f"    Mean Delta:        {xv.get('mean_delta', 0):.4f}")

        print(f"\n  ---- Diagnostic Flags ----")
        print(f"    Loop Suspected:       {diag.get('loop_suspected_rate', 0):.3f}")
        print(f"    Ignition Saturated:   {diag.get('ignition_saturated_rate', 0):.3f}")
        print(f"    Sensory Deprived:     {diag.get('sensory_deprived_rate', 0):.3f}")
        print(f"    Correction Gain:      {diag.get('mean_correction_gain', 0):.3f}")
        print(f"    Trend Aligned:        {diag.get('trend_aligned_rate', 0):.3f}")

        if diag.get('severe_issues'):
            print(f"\n  ---- SEVERE ISSUES ----")
            for issue in diag['severe_issues']:
                print(f"    ! {issue}")


class EnhancedVirtualWorld:
    def __init__(self, size=10.0, seed=42):
        import random
        self.size = size
        self.rng = random.Random(seed)
        self.agent_pos = np.array([size / 2.0, size / 2.0], dtype=np.float32)
        self.step_count = 0
        self.objects = {}
        self.object_id_counter = 0
        self._spawn_objects()
        self.success_count = 0

    def _spawn_objects(self):
        self.objects.clear()
        n_targets = 12
        while len(self.objects) < n_targets:
            x = self.rng.uniform(0.5, self.size - 0.5)
            y = self.rng.uniform(0.5, self.size - 0.5)
            pos = np.array([x, y], dtype=np.float32)
            if np.linalg.norm(pos - self.agent_pos) < 2.0:
                continue
            obj_type = self.rng.choice(["food", "danger", "object", "wall", "friend"])
            self.object_id_counter += 1
            self.objects[self.object_id_counter] = {
                "pos": pos.copy(),
                "type": obj_type,
                "size": self.rng.uniform(0.3, 0.9),
            }

    def get_sensory_package(self):
        visual = np.zeros(20, dtype=np.float32)
        auditory = np.zeros(14, dtype=np.float32)
        tactile = np.zeros(12, dtype=np.float32)
        vestibular = np.zeros(10, dtype=np.float32)
        place_cells = np.zeros(25, dtype=np.float32)
        olfactory = np.zeros(8, dtype=np.float32)

        for obj_id, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            direction_x = (obj["pos"][0] - self.agent_pos[0]) / max(0.01, dist)
            direction_y = (obj["pos"][1] - self.agent_pos[1]) / max(0.01, dist)

            if dist < 0.7:
                idx = min(obj_id % 12, 11)
                tactile[idx] = 1.0
                if obj["type"] == "danger":
                    tactile[(idx + 1) % 12] = 0.6
            elif dist < 3.5:
                idx = obj_id % 20
                visual[idx] = max(0.0, 1.0 - dist / 3.5)
                visual[(idx + 5) % 20] = max(0.0, (1.0 - dist / 3.5) * 0.4)
            elif dist < 7.0:
                idx = obj_id % 14
                auditory[idx] = max(0.0, 1.0 - dist / 7.0)

            if dist < 5.0 and obj["type"] == "food":
                olfactory[min(obj_id % 8, 7)] = max(0.0, 1.0 - dist / 5.0)

        px = int(self.agent_pos[0] / self.size * 5) % 5
        py = int(self.agent_pos[1] / self.size * 5) % 5
        place_cells[px * 5 + py] = 1.0
        place_cells[(px * 5 + py + 1) % 25] = 0.3

        vx = float(np.sin(self.agent_pos[0] * 0.5 + self.rng.uniform(-0.2, 0.2)))
        vy = float(np.cos(self.agent_pos[1] * 0.5 + self.rng.uniform(-0.2, 0.2)))
        vestibular[0] = vx
        vestibular[1] = vy
        vestibular[2] = self.rng.uniform(-0.3, 0.3)

        return {
            "visual": visual.tolist(),
            "auditory": auditory.tolist(),
            "tactile": tactile.tolist(),
            "vestibular": vestibular.tolist(),
            "place_cells": place_cells.tolist(),
            "olfactory": olfactory.tolist(),
        }

    def describe(self):
        nearby = []
        for obj_id, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            if dist < 5.0:
                direction = "ahead" if obj["pos"][1] > self.agent_pos[1] else "behind"
                nearby.append(f"{obj['type']}_{direction}")
        return " ".join(nearby[:12]) if nearby else "empty"

    def tick(self):
        self.step_count += 1
        for obj_id, obj in self.objects.items():
            obj["pos"] += np.array([
                self.rng.uniform(-0.15, 0.15),
                self.rng.uniform(-0.15, 0.15)
            ], dtype=np.float32)
            obj["pos"] = np.clip(obj["pos"], 0.0, self.size)

            if self.step_count % 200 == 0 and self.rng.random() < 0.05:
                obj["type"] = self.rng.choice(["food", "danger", "object", "wall", "friend"])

    def try_interact(self, motor_output):
        if not motor_output or len(motor_output) < 2:
            return 0.0
        dx = (float(motor_output[0] % 9) - 4.0) * 0.4
        dy = (float(motor_output[1] % 9) - 4.0) * 0.4
        self.agent_pos += np.array([dx, dy], dtype=np.float32)
        self.agent_pos = np.clip(self.agent_pos, 0.0, self.size)

        reward = 0.0
        for obj_id, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            if dist < 0.6:
                if obj["type"] == "food":
                    reward += 1.0
                    self.success_count += 1
                elif obj["type"] == "danger":
                    reward -= 0.5
                elif obj["type"] == "friend":
                    reward += 0.3
        return reward


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, world_class=None):
        self.config = config
        self.world_class = world_class if world_class is not None else EnhancedVirtualWorld

    def _build_concepts(self) -> List[str]:
        return [
            "self", "world", "move", "see", "eat",
            "good", "bad", "near", "far", "red",
            "blue", "green", "yellow", "up", "down",
            "left", "right", "object", "food", "wall",
            "empty", "reward", "danger", "safe", "want",
            "think", "feel", "know", "I", "you",
            "big", "small", "fast", "slow", "hot",
            "cold", "light", "dark", "open", "close",
            "push", "pull", "give", "take", "make",
            "break", "start", "stop", "before", "after",
            "same", "different", "more", "less", "all",
            "some", "none", "here", "there", "now",
            "friend", "smell", "palpitation", "hungry", "tired",
            "curious", "afraid", "happy", "ready", "wait",
        ]

    def run_seed(self, seed: int) -> SeedResult:
        os.environ["OMP_NUM_THREADS"] = str(self.config.omp_num_threads)

        import random
        random.seed(seed)
        np.random.seed(seed)

        brain = core_cpp.CorticalBrain(self.config.n_neurons, self._build_concepts())
        brain.set_seed(seed)

        calibrator = ConsciousnessCalibrator(window_size=self.config.correction_window_size)
        correction_engine = None
        if self.config.enable_correction_engine:
            correction_engine = PhiCorrectionEngine(
                window_size=self.config.correction_window_size
            )

        world = self.world_class(self.config.world_size, seed)
        result = SeedResult(seed=seed)

        t_start = time.perf_counter()

        sp = world.get_sensory_package()
        combined_visual = sp["visual"].copy()
        combined_visual.extend(sp.get("olfactory", []))
        brain.inject_multi_modal(combined_visual, sp["auditory"],
                                  sp["tactile"], sp["vestibular"],
                                  sp["place_cells"])
        brain.inject_text(world.describe())

        step = 0
        loop_suspected_steps = 0
        ignition_saturated_steps = 0
        sensory_deprived_steps = 0
        correction_gain_history = []
        trend_aligned_count = 0
        trend_check_count = 0

        injection_interval = self.config.sensory_injection_interval

        for episode in range(self.config.n_episodes):
            episode_success = 0
            for s in range(self.config.n_steps_per_episode):
                step += 1
                reward = 0.0

                if step % injection_interval == 0:
                    sp = world.get_sensory_package()
                    combined_visual = sp["visual"].copy()
                    combined_visual.extend(sp.get("olfactory", []))
                    brain.inject_multi_modal(combined_visual, sp["auditory"],
                                              sp["tactile"], sp["vestibular"],
                                              sp["place_cells"])
                    brain.inject_text(world.describe())
                    world.tick()

                motor = brain.read_motor_output()
                reward = world.try_interact(motor)
                if reward > 0:
                    episode_success += 1
                brain.step(reward)

                cs = brain.read_consciousness()

                raw_phi = cs.phi
                raw_ignition = cs.global_ignition
                raw_self_error = cs.self_prediction_error
                vividness = brain.get_perceptual_vividness()
                fp_salience = brain.get_first_person_salience()
                goal_prog = brain.get_goal_progress()

                result.phi_history.append(raw_phi)
                result.ignition_history.append(raw_ignition)
                result.self_pred_errors.append(raw_self_error)
                result.vividness_history.append(vividness)
                result.fp_salience_history.append(fp_salience)
                result.goal_progress.append(goal_prog)

                if correction_engine is not None:
                    behavioral_score = (float(episode_success) / max(1.0, float(s + 1))
                                        + goal_prog) / 2.0
                    correction_engine.add_observation(
                        phi=raw_phi,
                        behavioral_score=behavioral_score,
                        ignition=raw_ignition,
                        vividness=vividness,
                    )

                    rec = correction_engine.get_recommendation()
                    if rec.get('status') == 'divergence_detected':
                        trend_check_count += 1
                        if rec.get('trend_aligned', True):
                            trend_aligned_count += 1

                    correction_gain_history.append(correction_engine.get_correction_gain())

                    is_loop, _ = check_self_referential_loop(raw_self_error, vividness)
                    if is_loop:
                        loop_suspected_steps += 1

                    if len(result.ignition_history) >= 20:
                        is_sat, _ = check_ignition_health(result.ignition_history[-20:])
                        if is_sat:
                            ignition_saturated_steps += 1

                if vividness < 0.02:
                    sensory_deprived_steps += 1

                if step % self.config.calibration_interval == 0 and step >= self.config.calibration_start_step:
                    _, comps = compute_consciousness_from_cortical_brain(
                        brain, calibrator=calibrator, step=step
                    )
                    result.consciousness_levels.append(comps['consciousness_level'])

                if step % 50 == 0:
                    text = brain.read_output_text()
                    result.output_texts.append(text)

            result.success_history.append(episode_success)
            result.total_successes += episode_success

        result.elapsed_seconds = time.perf_counter() - t_start
        result.total_steps = step

        cs = brain.read_consciousness()
        result.final_phi = cs.phi
        consciousness_raw, _ = compute_consciousness_from_cortical_brain(
            brain, calibrator=calibrator, step=step
        )
        result.final_consciousness = consciousness_raw

        total_readings = max(1, len(result.phi_history))
        result.loop_suspected_rate = loop_suspected_steps / total_readings
        result.ignition_saturated_rate = ignition_saturated_steps / total_readings
        result.sensory_deprived_rate = sensory_deprived_steps / total_readings

        if correction_engine is not None:
            stats = correction_engine.get_stats()
            result.final_phi_corrected = stats['phi_mean']
            result.correction_engine_stats = {
                'correction_gain': correction_engine.get_correction_gain(),
                'pearson_r': stats['pearson_r'],
                'spearman_rho': stats['spearman_rho'],
                'p_value': stats['p_value'],
                'divergence_detected': stats['divergence_detected'],
                'divergence_count': stats['divergence_count'],
                'trend_aligned': stats['trend_aligned'],
                'phi_slope': stats['phi_slope'],
                'behavior_slope': stats['behavior_slope'],
            }

        if calibrator.step_count > 0:
            result.calibrator_diagnostics = calibrator.get_diagnostics()

        try:
            xval = cross_validate_consciousness(brain, calibrator=calibrator, step=step)
            result.calibrator_diagnostics['cross_validation'] = {
                'aligned': xval['aligned'],
                'delta': xval['delta'],
                'diagnosis': xval['diagnosis'],
            }
        except Exception:
            pass

        return result

    def run_all(self, progress_callback=None) -> ExperimentReport:
        report = ExperimentReport(config=self.config)

        for i in range(self.config.n_seeds):
            seed = self.config.seed_offset + i
            if progress_callback:
                progress_callback(i, self.config.n_seeds, seed)

            print(f"\n  Running seed {seed} ({i+1}/{self.config.n_seeds})...")
            result = self.run_seed(seed)
            report.seed_results.append(result)
            summary = result.compute_summary()
            print(f"    Phi={summary.get('mean_phi',0):.3f}  "
                  f"Ign={summary.get('mean_ignition',0):.3f}  "
                  f"Succ={summary.get('success_rate',0):.2%}  "
                  f"Loop={result.loop_suspected_rate:.2%}  "
                  f"Sat={result.ignition_saturated_rate:.2%}")

        self._compute_aggregate(report)
        self._compute_stability(report)
        self._compute_correlations(report)
        self._compute_diagnostics(report)

        return report

    def _compute_aggregate(self, report: ExperimentReport):
        summaries = [r.compute_summary() for r in report.seed_results]
        if not summaries:
            return

        keys = [
            'mean_phi', 'mean_phi_corrected',
            'mean_ignition', 'mean_ignition_corrected',
            'mean_vividness', 'mean_fp_salience',
            'mean_consciousness', 'mean_consciousness_corrected',
            'success_rate',
            'mean_self_pred_error', 'mean_self_pred_error_corrected',
            'steps_per_second', 'correction_gain',
        ]
        agg = {}

        for key in keys:
            values = [s.get(key) for s in summaries if key in s and s[key] is not None]
            if not values:
                continue
            arr = np.array(values)
            n = len(arr)
            agg[f'{key}_mean'] = float(np.mean(arr))
            agg[f'{key}_std'] = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            agg[f'{key}_min'] = float(np.min(arr))
            agg[f'{key}_max'] = float(np.max(arr))

            if n > 1:
                from scipy import stats
                arr_std = float(np.std(arr, ddof=1))
                if arr_std == 0.0:
                    agg[f'{key}_ci_low'] = agg[f'{key}_mean']
                    agg[f'{key}_ci_high'] = agg[f'{key}_mean']
                else:
                    ci = stats.t.interval(0.95, df=n-1, loc=np.mean(arr),
                                           scale=stats.sem(arr))
                    agg[f'{key}_ci_low'] = float(ci[0])
                    agg[f'{key}_ci_high'] = float(ci[1])
            else:
                agg[f'{key}_ci_low'] = agg[f'{key}_mean']
                agg[f'{key}_ci_high'] = agg[f'{key}_mean']

        report.aggregate = agg

    def _compute_stability(self, report: ExperimentReport):
        summaries = [r.compute_summary() for r in report.seed_results]
        if len(summaries) < 2:
            report.phi_stability = {'reproducibility_score': 0.0}
            return

        phi_values = [s.get('mean_phi', 0) for s in summaries]
        phi_corr_values = [s.get('mean_phi_corrected', s.get('mean_phi', 0)) for s in summaries]
        consc_values = [s.get('mean_consciousness', 0) for s in summaries]

        phi_arr = np.array(phi_values)
        phi_corr_arr = np.array(phi_corr_values)
        consc_arr = np.array(consc_values)

        phi_cov = float(np.std(phi_arr, ddof=1) / max(1e-6, np.mean(phi_arr)))
        phi_corr_cov = float(np.std(phi_corr_arr, ddof=1) / max(1e-6, np.mean(phi_corr_arr)))
        consc_cov = float(np.std(consc_arr, ddof=1) / max(1e-6, np.mean(consc_arr)))

        reproducibility = 1.0 / (1.0 + phi_corr_cov)

        report.phi_stability = {
            'phi_cov': phi_cov,
            'phi_corrected_cov': phi_corr_cov,
            'consciousness_cov': consc_cov,
            'reproducibility_score': reproducibility,
            'phi_range': float(np.max(phi_arr) - np.min(phi_arr)),
            'phi_corrected_range': float(np.max(phi_corr_arr) - np.min(phi_corr_arr)),
        }

    def _compute_correlations(self, report: ExperimentReport):
        all_phis = []
        all_phis_corrected = []
        all_successes = []
        for r in report.seed_results:
            summary = r.compute_summary()
            all_phis.append(summary.get('mean_phi', 0))
            all_phis_corrected.append(summary.get('mean_phi_corrected', summary.get('mean_phi', 0)))
            all_successes.append(summary.get('success_rate', 0))

        if len(all_phis) < 3:
            report.correlation_analysis = {
                'pearson_r': 0, 'pearson_r_corrected': 0,
                'interpretation': 'insufficient data (<3 seeds)'
            }
            return

        from scipy import stats
        r, p = stats.pearsonr(all_phis, all_successes)
        r_corr, p_corr = stats.pearsonr(all_phis_corrected, all_successes)
        rho, p_spearman = stats.spearmanr(all_phis, all_successes)

        if abs(r_corr) < 0.3:
            interp = "弱相关"
        elif abs(r_corr) < 0.6:
            interp = "中等相关"
        else:
            interp = "强相关"

        sig_str = "显著" if p_corr < 0.05 else "不显著"
        interp += f" ({sig_str}, n={len(all_phis)}, p_corr={p_corr:.4f})"

        report.correlation_analysis = {
            'pearson_r': float(r),
            'pearson_r_corrected': float(r_corr),
            'p_value': float(p),
            'p_value_corrected': float(p_corr),
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
            'n_samples': len(all_phis),
            'interpretation': interp,
        }

    def _compute_diagnostics(self, report: ExperimentReport):
        if not report.seed_results:
            report.diagnostic_summary = {}
            return

        loop_rates = [r.loop_suspected_rate for r in report.seed_results]
        ign_sat_rates = [r.ignition_saturated_rate for r in report.seed_results]
        sens_dep_rates = [r.sensory_deprived_rate for r in report.seed_results]
        correction_gains = [r.correction_engine_stats.get('correction_gain', 0)
                            for r in report.seed_results]

        trend_aligned_flags = [r.correction_engine_stats.get('trend_aligned', True)
                               for r in report.seed_results]
        trend_aligned_rate = sum(trend_aligned_flags) / max(1, len(trend_aligned_flags))

        all_ignitions = []
        for r in report.seed_results:
            all_ignitions.extend(r.ignition_history[-500:] if r.ignition_history else [])
        mean_ignition_variance = float(np.var(all_ignitions)) if len(all_ignitions) > 1 else 0.0

        report.diagnostic_summary = {
            'loop_suspected_rate': float(np.mean(loop_rates)),
            'loop_suspected_rate_std': float(np.std(loop_rates, ddof=1)) if len(loop_rates) > 1 else 0.0,
            'ignition_saturated_rate': float(np.mean(ign_sat_rates)),
            'ignition_saturated_rate_std': float(np.std(ign_sat_rates, ddof=1)) if len(ign_sat_rates) > 1 else 0.0,
            'sensory_deprived_rate': float(np.mean(sens_dep_rates)),
            'sensory_deprived_rate_std': float(np.std(sens_dep_rates, ddof=1)) if len(sens_dep_rates) > 1 else 0.0,
            'mean_correction_gain': float(np.mean(correction_gains)),
            'mean_ignition_variance': mean_ignition_variance,
            'trend_aligned_rate': trend_aligned_rate,
        }

        xv_aligned = 0
        xv_deltas = []
        for r in report.seed_results:
            xv = r.calibrator_diagnostics.get('cross_validation', {})
            if xv.get('aligned', False):
                xv_aligned += 1
            delta = xv.get('delta', None)
            if delta is not None:
                xv_deltas.append(delta)
        n_seeds = max(1, len(report.seed_results))
        report.diagnostic_summary['cross_validation'] = {
            'aligned_ratio': xv_aligned / n_seeds,
            'mean_delta': float(np.mean(xv_deltas)) if xv_deltas else 0.0,
            'max_delta': float(np.max(xv_deltas)) if xv_deltas else 0.0,
        }

        severe_issues = []
        if report.diagnostic_summary['loop_suspected_rate'] > 0.3:
            severe_issues.append("SELF_REFERENTIAL_LOOP: self_model 在无感官输入下自我预测")
        if report.diagnostic_summary['ignition_saturated_rate'] > 0.5:
            severe_issues.append("GWT_SATURATED: 全局工作空间缺乏有效竞争")
        if report.diagnostic_summary['sensory_deprived_rate'] > 0.3:
            severe_issues.append("SENSORY_DEPRIVED: 感官注入频率不足")
        if mean_ignition_variance < 0.0005:
            severe_issues.append("IGNITION_ZERO_VARIANCE: ignition 值单一无变化")
        if report.diagnostic_summary.get('cross_validation', {}).get('aligned_ratio', 1.0) < 0.5:
            severe_issues.append("CROSS_VALIDATION_MISMATCH: C++/Python consciousness 计算不一致")

        report.diagnostic_summary['severe_issues'] = severe_issues


def run_standard_experiment(world_class=None) -> ExperimentReport:
    config = ExperimentConfig(
        n_neurons=32000,
        n_episodes=300,
        n_steps_per_episode=50,
        n_seeds=10,
        seed_offset=42,
        enable_correction_engine=True,
        sensory_injection_interval=3,
    )
    runner = ExperimentRunner(config, world_class=world_class)
    report = runner.run_all()
    report.print_summary()
    report.save(os.path.join(config.output_dir, 'standard_experiment_v6.json'))
    return report


if __name__ == '__main__':
    run_standard_experiment()