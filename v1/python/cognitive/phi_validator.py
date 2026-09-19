"""
Phi-Behavior 校准验证器 v4

在 Phi（信息整合量）与可观测行为表现之间建立联系，
验证意识水平指标是否真正反映系统的有效功能。

v4 增强:
  - 在线反馈闭环: 检测到脱钩时自动建议校正参数
  - 趋势分析: 检测 Phi/Behavior 趋势是否一致
  - 早期预警: 在完全脱钩之前发出警告
  - 自适应窗口: 根据方差动态调整相关窗口大小
  - 与 ConsciousnessCalibrator 联动
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, List, Optional, Deque


def _pearson(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx, my = x.mean(), y.mean()
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 1.0
    r = np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy)
    r = np.clip(r, -1.0, 1.0)
    z = 0.5 * np.log((1 + r + 1e-9) / (1 - r + 1e-9))
    se = 1.0 / np.sqrt(n - 3)
    z_stat = abs(z) / se if se > 0 else 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(z_stat))
    return float(r), float(p)


def _spearman(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _detect_divergence_trend(history: Deque[float], window: int = 10) -> bool:
    if len(history) < window + 5:
        return False
    early = np.mean([history[i] for i in range(len(history) - window - 5, len(history) - window)])
    late = np.mean([history[i] for i in range(len(history) - window, len(history))])
    return abs(late - early) > 0.15


def _compute_slope(history: List[float]) -> float:
    if len(history) < 5:
        return 0.0
    n = len(history)
    x = np.arange(n, dtype=float)
    y = np.array(history, dtype=float)
    mx, my = x.mean(), y.mean()
    slope = np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2) if np.sum((x - mx) ** 2) > 0 else 0.0
    return slope / max(0.001, abs(my) if abs(my) > 0.001 else 1.0)


class PhiCorrectionEngine:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.phi_history: Deque[float] = deque(maxlen=window_size)
        self.behavior_history: Deque[float] = deque(maxlen=window_size)
        self.correlation_history: Deque[float] = deque(maxlen=window_size)
        self.ignition_history: Deque[float] = deque(maxlen=window_size)
        self.vividness_history: Deque[float] = deque(maxlen=window_size)
        self.divergence_detected = False
        self.divergence_count = 0
        self.last_pearson_r = 0.0
        self.last_spearman_rho = 0.0
        self.last_p_value = 1.0
        self.correction_gain = 0.0
        self.adaptive_window = window_size
        self.phi_slope = 0.0
        self.behavior_slope = 0.0
        self.trend_aligned = True

    def add_observation(self, phi: float, behavioral_score: float,
                         ignition: float = 0.0, vividness: float = 0.0):
        self.phi_history.append(float(np.clip(phi, 0.0, 1.0)))
        self.behavior_history.append(float(np.clip(behavioral_score, 0.0, 1.0)))
        self.ignition_history.append(float(np.clip(ignition, 0.0, 1.0)))
        self.vividness_history.append(float(np.clip(vividness, 0.0, 1.0)))

        if len(self.phi_history) >= 10:
            phi_var = float(np.var(list(self.phi_history)))
            behavior_var = float(np.var(list(self.behavior_history)))
            if phi_var < 0.001 and behavior_var < 0.001:
                self.adaptive_window = min(200, self.window_size * 2)
            elif phi_var > 0.05 or behavior_var > 0.05:
                self.adaptive_window = max(20, self.window_size // 2)
            else:
                self.adaptive_window = self.window_size

        if len(self.phi_history) >= 10:
            r, p = _pearson(list(self.phi_history)[-self.adaptive_window:],
                             list(self.behavior_history)[-self.adaptive_window:])
            self.last_pearson_r = r
            self.last_p_value = p

            rho, _ = _spearman(list(self.phi_history)[-self.adaptive_window:],
                               list(self.behavior_history)[-self.adaptive_window:])
            self.last_spearman_rho = rho

            self.correlation_history.append(r)

            self.phi_slope = _compute_slope(list(self.phi_history)[-20:])
            self.behavior_slope = _compute_slope(list(self.behavior_history)[-20:])
            self.trend_aligned = (self.phi_slope * self.behavior_slope) >= 0

            phi_list = list(self.phi_history)
            behavior_list = list(self.behavior_history)

            recent_phi = phi_list[-min(15, len(phi_list)):]
            recent_behave = behavior_list[-min(15, len(behavior_list)):]
            phi_trend = _compute_slope(recent_phi)
            behave_trend = _compute_slope(recent_behave)

            phi_diverging = _detect_divergence_trend(self.phi_history)
            behave_diverging = _detect_divergence_trend(self.behavior_history)

            self.divergence_detected = (
                (r < 0.1 and r > -100)
                or (phi_trend * behave_trend < -0.01)
                or (phi_diverging != behave_diverging and len(self.phi_history) > 30)
            )

            if self.divergence_detected:
                self.divergence_count += 1
            else:
                self.divergence_count = max(0, self.divergence_count - 1)

            if self.divergence_count > 5:
                ig_var = float(np.var(list(self.ignition_history)[-20:])) if len(self.ignition_history) >= 20 else 0.0
                vd_mean = float(np.mean(list(self.vividness_history)[-20:])) if len(self.vividness_history) >= 20 else 0.0

                if ig_var < 0.0005:
                    self.correction_gain = min(1.0, self.correction_gain + 0.08)
                elif vd_mean < 0.05:
                    self.correction_gain = min(1.0, self.correction_gain + 0.10)
                else:
                    self.correction_gain = min(1.0, self.correction_gain + 0.04)

    def get_correction_gain(self) -> float:
        return self.correction_gain

    def get_recommendation(self) -> Dict:
        if not self.divergence_detected:
            return {
                "status": "normal",
                "pearson_r": self.last_pearson_r,
                "spearman_rho": self.last_spearman_rho,
                "p_value": self.last_p_value,
                "correction_gain": self.correction_gain,
                "trend_aligned": self.trend_aligned,
                "phi_slope": self.phi_slope,
                "behavior_slope": self.behavior_slope,
            }

        ig_var = float(np.var(list(self.ignition_history)[-20:])) if len(self.ignition_history) >= 20 else 0.0
        vd_mean = float(np.mean(list(self.vividness_history)[-20:])) if len(self.vividness_history) >= 20 else 0.0

        recommendations = []
        if ig_var < 0.0005:
            recommendations.append("GWT_IGNITION_SATURATED: Increase lateral inhibition or add winner fatigue")
        if vd_mean < 0.05:
            recommendations.append("SENSORY_DEPRIVED: Increase sensory injection frequency or amplitude")
        if self.last_pearson_r < 0.1:
            recommendations.append("PHI_BEHAVIOR_DECOUPLED")
        if not self.trend_aligned:
            recommendations.append("TREND_MISALIGNED: Phi and behavior move in opposite directions")

        return {
            "status": "divergence_detected",
            "divergence_count": self.divergence_count,
            "pearson_r": self.last_pearson_r,
            "spearman_rho": self.last_spearman_rho,
            "p_value": self.last_p_value,
            "correction_gain": self.correction_gain,
            "ignition_variance": ig_var,
            "vividness_mean": vd_mean,
            "trend_aligned": self.trend_aligned,
            "phi_slope": self.phi_slope,
            "behavior_slope": self.behavior_slope,
            "recommendations": recommendations,
        }

    def get_stats(self) -> Dict:
        phi_arr = np.array(list(self.phi_history)) if self.phi_history else np.array([0.0])
        behave_arr = np.array(list(self.behavior_history)) if self.behavior_history else np.array([0.0])

        return {
            "n": len(self.phi_history),
            "phi_mean": float(phi_arr.mean()),
            "phi_std": float(phi_arr.std(ddof=1)) if len(phi_arr) > 1 else 0.0,
            "behavior_mean": float(behave_arr.mean()),
            "behavior_std": float(behave_arr.std(ddof=1)) if len(behave_arr) > 1 else 0.0,
            "pearson_r": self.last_pearson_r,
            "spearman_rho": self.last_spearman_rho,
            "p_value": self.last_p_value,
            "correction_gain": self.correction_gain,
            "divergence_detected": self.divergence_detected,
            "divergence_count": self.divergence_count,
            "adaptive_window": self.adaptive_window,
            "trend_aligned": self.trend_aligned,
            "phi_slope": self.phi_slope,
            "behavior_slope": self.behavior_slope,
        }


class PhiBehaviorCalibration:
    def __init__(self, window_size: int = 200):
        self.phi_history: Deque[float] = deque(maxlen=window_size)
        self.behavior_history: Deque[float] = deque(maxlen=window_size)
        self.level_history: Deque[float] = deque(maxlen=window_size)
        self.session_stats: Dict = {"n": 0}

    def update(self, phi: float, behavior_score: float, conscious_level: float):
        self.phi_history.append(float(phi))
        self.behavior_history.append(float(behavior_score))
        self.level_history.append(float(conscious_level))

    def get_phi_behavior_correlation(self) -> Tuple[float, float, float]:
        n = len(self.phi_history)
        if n < 5:
            return 0.0, 0.0, 1.0
        r, p = _pearson(list(self.phi_history), list(self.behavior_history))
        rho, _ = _spearman(list(self.phi_history), list(self.behavior_history))
        return r, rho, p

    def get_stats(self) -> Dict:
        return {
            "n": len(self.phi_history),
            "phi_mean": float(np.mean(list(self.phi_history))) if self.phi_history else 0.0,
            "behavior_mean": float(np.mean(list(self.behavior_history))) if self.behavior_history else 0.0,
            "level_mean": float(np.mean(list(self.level_history))) if self.level_history else 0.0,
        }


def check_self_referential_loop(
    self_prediction_error: float,
    perceptual_vividness: float,
    error_threshold: float = 0.01,
    vividness_threshold: float = 0.10
) -> Tuple[bool, str]:
    if self_prediction_error < error_threshold and perceptual_vividness < vividness_threshold:
        return True, (
            f"Self-referential loop suspected: "
            f"self_prediction_error={self_prediction_error:.6f} < {error_threshold}, "
            f"vividness={perceptual_vividness:.4f} < {vividness_threshold}. "
            f"Model predicts itself too well without sensory grounding."
        )
    return False, "Normal: Prediction error and sensory engagement in healthy range."


def check_ignition_health(ignition_values: List[float]) -> Tuple[bool, str]:
    if len(ignition_values) < 20:
        return False, "Insufficient data for ignition health check."
    recent = ignition_values[-20:]
    variance = float(np.var(recent))
    if variance < 0.0005:
        return True, (
            f"GWT ignition saturated: variance={variance:.8f}. "
            f"Competition mechanism not functioning — check lateral inhibition strength."
        )
    return False, f"Ignition healthy: variance={variance:.6f}."


def format_report(stats: Dict, recommendations: Dict = None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Phi-Behavior Calibration Report")
    lines.append("=" * 60)
    lines.append(f"Observations:      {stats.get('n', 0)}")
    lines.append(f"Phi mean:          {stats.get('phi_mean', 0):.4f} ± {stats.get('phi_std', 0):.4f}")
    lines.append(f"Behavior mean:     {stats.get('behavior_mean', 0):.4f} ± {stats.get('behavior_std', 0):.4f}")
    lines.append(f"Pearson r:         {stats.get('pearson_r', 0):+.4f}")
    lines.append(f"Spearman ρ:        {stats.get('spearman_rho', 0):+.4f}")
    lines.append(f"p-value:           {stats.get('p_value', 1):.4f}")
    lines.append(f"Correction gain:   {stats.get('correction_gain', 0):.4f}")
    lines.append(f"Trend aligned:     {stats.get('trend_aligned', True)}")
    lines.append(f"Adaptive window:   {stats.get('adaptive_window', 'N/A')}")
    lines.append("-" * 60)

    if stats.get('divergence_detected', False):
        lines.append("WARNING: Phi-Behavior divergence detected!")
        if recommendations:
            lines.append(f"Status: {recommendations.get('status', 'unknown')}")
            for rec in recommendations.get('recommendations', []):
                lines.append(f"  → {rec}")

    return "\n".join(lines)


class PhiValidator:
    """集成验证器：统一管理 Phi 校正引擎、校准器和交叉验证。"""

    def __init__(self, window_size: int = 300):
        self.correction_engine = PhiCorrectionEngine(window_size)
        self.calibration = PhiBehaviorCalibration(window_size)
        self.n_steps = 0
        self.health_flags = {
            "ignition_saturated": False,
            "loop_detected": False,
            "sensory_deprived": False,
            "phi_behavior_decoupled": False,
        }

    def step(self, phi, behavioral_score, ignition, vividness,
             self_prediction_error, consciousness_level):
        self.n_steps += 1
        self.correction_engine.add_observation(phi, behavioral_score, ignition, vividness)
        self.calibration.update(phi, behavioral_score, consciousness_level)

        loop, _ = check_self_referential_loop(self_prediction_error, vividness)
        self.health_flags["loop_detected"] = loop

        if len(self.correction_engine.ignition_history) >= 20:
            ign_list = list(self.correction_engine.ignition_history)[-20:]
            is_sat, _ = check_ignition_health(ign_list)
            self.health_flags["ignition_saturated"] = is_sat

        if vividness < 0.02:
            self.health_flags["sensory_deprived"] = True
        elif vividness > 0.06:
            self.health_flags["sensory_deprived"] = False

        self.health_flags["phi_behavior_decoupled"] = (
            self.correction_engine.divergence_detected)

    def get_report(self):
        stats = self.correction_engine.get_stats()
        rec = self.correction_engine.get_recommendation()
        r, rho, p = self.calibration.get_phi_behavior_correlation()

        return {
            "n_steps": self.n_steps,
            "phi_mean": stats["phi_mean"],
            "phi_std": stats["phi_std"],
            "behavior_mean": stats["behavior_mean"],
            "behavior_std": stats["behavior_std"],
            "pearson_r": r,
            "spearman_rho": rho,
            "p_value": p,
            "correction_gain": self.correction_engine.correction_gain,
            "trend_aligned": self.correction_engine.trend_aligned,
            "health_flags": dict(self.health_flags),
            "recommendations": rec.get("recommendations", []),
        }

    def print_report(self):
        r = self.get_report()
        print("\n" + "=" * 60)
        print("SNA Integrated PhiValidator v4 Report")
        print("=" * 60)
        print(f"  Steps:              {r['n_steps']}")
        print(f"  Phi:                {r['phi_mean']:.4f} ± {r['phi_std']:.4f}")
        print(f"  Behavior:           {r['behavior_mean']:.4f} ± {r['behavior_std']:.4f}")
        print(f"  Pearson r:          {r['pearson_r']:+.4f}")
        print(f"  Spearman ρ:         {r['spearman_rho']:+.4f}")
        print(f"  Correction gain:    {r['correction_gain']:.4f}")
        print(f"  Trend aligned:      {r['trend_aligned']}")
        print(f"  ── Health Flags ──")
        for flag, val in r['health_flags'].items():
            status = "WARN" if val else "OK"
            print(f"    {flag}: {status}")
        for rec in r['recommendations']:
            print(f"  → {rec}")


def validate_cortical_brain(brain, n_steps=200):
    import core_cpp
    from .unified_consciousness import (
        compute_consciousness_from_cortical_brain,
        ConsciousnessCalibrator,
        cross_validate_consciousness,
        compute_consciousness_diagnostic_report,
    )

    calibrator = ConsciousnessCalibrator()
    validator = PhiValidator()

    phi_history = []
    ignition_history = []
    vividness_history = []
    fp_history = []
    consciousness_levels = []
    behavioral_scores = []
    all_self_errors = []

    for step in range(n_steps):
        brain.step(0.0)
        cs = brain.read_consciousness()
        vivid = brain.get_perceptual_vividness()
        fp = brain.get_first_person_salience()

        phi_history.append(cs.phi)
        ignition_history.append(cs.global_ignition)
        vividness_history.append(vivid)
        fp_history.append(fp)
        all_self_errors.append(cs.self_prediction_error)

        level, comps = compute_consciousness_from_cortical_brain(
            brain, calibrator=calibrator, step=step)
        consciousness_levels.append(level)

        behavior = brain.get_success_rate() if hasattr(brain, 'get_success_rate') else (
            (1.0 - cs.self_prediction_error) * 0.4 + vivid * 0.3
            + (cs.global_ignition if cs.global_ignition < 0.95 else 0.5) * 0.3)
        behavioral_scores.append(behavior)

        validator.step(cs.phi, behavior, cs.global_ignition, vivid,
                       cs.self_prediction_error, level)

    tail = min(100, n_steps)
    avg_phi = float(np.mean(phi_history[-tail:]))
    avg_ignition = float(np.mean(ignition_history[-tail:]))
    avg_vividness = float(np.mean(vividness_history[-tail:]))
    avg_fp = float(np.mean(fp_history[-tail:]))
    avg_consciousness = float(np.mean(consciousness_levels[-tail:]))
    avg_self_error = float(np.mean(all_self_errors[-tail:]))
    avg_behavior = float(np.mean(behavioral_scores[-tail:]))

    cross = cross_validate_consciousness(brain, calibrator, step=n_steps)

    if avg_vividness > 0.001:
        corrected_avg_phi = avg_phi * min(1.0, avg_behavior / max(0.001, avg_vividness))
    else:
        corrected_avg_phi = avg_phi * 0.3

    summary = {
        "avg_phi": avg_phi,
        "corrected_avg_phi": corrected_avg_phi,
        "avg_ignition": avg_ignition,
        "avg_vividness": avg_vividness,
        "avg_fp_salience": avg_fp,
        "avg_consciousness": avg_consciousness,
        "avg_self_pred_error": avg_self_error,
        "avg_behavioral_score": avg_behavior,
        "cross_validation": cross,
        "flags": {
            "ignition_saturated": avg_ignition > 0.95,
            "loop_risk": avg_self_error < 0.01 and avg_vividness < 0.1,
            "sensory_deprived": avg_vividness < 0.02,
            "low_vividness": avg_vividness < 0.06,
            "phi_behavior_decoupled": validator.health_flags["phi_behavior_decoupled"],
        }
    }

    diagnostic = compute_consciousness_diagnostic_report(
        brain, calibrator=calibrator, step=n_steps)

    return validator, summary, diagnostic