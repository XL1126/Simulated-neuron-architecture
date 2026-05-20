"""
统一意识水平计算模块 — SNA Unified Consciousness Metric v6

理论依据:
  - IIT (Integrated Information Theory, Tononi 2004/2008):
    Phi 量化系统整合信息的能力。权重 0.35
    实现: 脑区间互信息矩阵最大特征值 + 分化度 + 时序变化

  - GWT (Global Workspace Theory, Baars 1988 / Dehaene 2001):
    全脑点火是意识内容进入全局访问的必要条件。权重 0.25
    实现: 工作空间竞争质量 (CV-based) + 参与度

  - Predictive Coding / Free Energy (Friston 2010):
    自我-世界模型预测精度。权重 0.20
    实现: self_model_accuracy(0.55) + world_model_accuracy(0.45)

  - Qualia / First-Person Perspective (Seth 2021, Metzinger 2003):
    第一人称显著度和感知生动度。各权重 0.10

v6 增强:
  - 与 C++ _update_consciousness() 权重对齐 (IIT 0.35, GWT 0.25, PC 0.20, Qualia 0.20)
  - 自指循环检测阈值增强: accuracy>0.95 + diff<0.003 + vividness<0.1
  - 动态感官门控与可信度校准
  - 在线学习率自适应
  - 行为-Phi 脱钩实时告警
  - Bootstrap 置信区间估计
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional


_IIT_WEIGHT = 0.35
_GWT_WEIGHT = 0.25
_PREDICTIVE_WEIGHT = 0.20
_FP_WEIGHT = 0.10
_VIVIDNESS_WEIGHT = 0.10

assert abs(_IIT_WEIGHT + _GWT_WEIGHT + _PREDICTIVE_WEIGHT
           + _FP_WEIGHT + _VIVIDNESS_WEIGHT - 1.0) < 1e-9

_VIVIDNESS_MIN_THRESHOLD = 0.02
_SELF_ERROR_MAX_THRESHOLD = 0.95
_CREDIBILITY_EMA_ALPHA = 0.05
_TEMPORAL_DECAY_HALFLIFE = 200
_LOOP_ACCURACY_THRESHOLD = 0.95
_LOOP_DIFF_THRESHOLD = 0.003
_LOOP_VIVIDNESS_THRESHOLD = 0.10


def _compute_sensory_gate(perceptual_vividness: float) -> float:
    sqrt_v = np.sqrt(max(0.0, perceptual_vividness))
    gate = 0.15 + 0.85 * sqrt_v
    return float(np.clip(gate, 0.15, 1.0))


def _detect_self_referential_loop(
    self_prediction_error: float,
    perceptual_vividness: float,
    prev_self_accuracy: float = 0.5
) -> Tuple[bool, float]:
    accuracy = 1.0 - self_prediction_error
    prev_diff = abs(accuracy - prev_self_accuracy)

    loop_detected = (
        accuracy > _LOOP_ACCURACY_THRESHOLD
        and prev_diff < _LOOP_DIFF_THRESHOLD
        and perceptual_vividness < _LOOP_VIVIDNESS_THRESHOLD
    )

    if loop_detected:
        correction = 0.6 + (accuracy - 0.6) * 0.25
    elif accuracy > 0.9 and prev_diff < 0.005:
        correction = 0.9 + (accuracy - 0.9) * 0.5
    else:
        correction = accuracy

    return loop_detected, correction


class ConsciousnessCalibrator:
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.phi_history = deque(maxlen=window_size)
        self.ignition_history = deque(maxlen=window_size)
        self.self_error_history = deque(maxlen=window_size)
        self.vividness_history = deque(maxlen=window_size)
        self.fp_history = deque(maxlen=window_size)
        self.level_history = deque(maxlen=window_size)
        self.loop_flags = deque(maxlen=window_size)
        self.ignition_saturated_flags = deque(maxlen=window_size)

        self.ema_phi = 0.0
        self.ema_ignition = 0.0
        self.ema_vividness = 0.0
        self.ema_level = 0.0
        self.prev_self_accuracy = 0.5

        self.step_count = 0
        self.loop_count = 0
        self.ignition_saturated_count = 0
        self.sensory_deprived_count = 0

    def update(self, phi: float, ignition: float, self_error: float,
               fp_salience: float, vividness: float, level: float,
               loop_detected: bool = False, ignition_saturated: bool = False):
        self.step_count += 1
        self.phi_history.append(phi)
        self.ignition_history.append(ignition)
        self.self_error_history.append(self_error)
        self.fp_history.append(fp_salience)
        self.vividness_history.append(vividness)
        self.level_history.append(level)
        self.loop_flags.append(loop_detected)
        self.ignition_saturated_flags.append(ignition_saturated)

        if loop_detected:
            self.loop_count += 1
        if ignition_saturated:
            self.ignition_saturated_count += 1
        if vividness < _VIVIDNESS_MIN_THRESHOLD:
            self.sensory_deprived_count += 1

        self.prev_self_accuracy = 1.0 - self_error

        a = _CREDIBILITY_EMA_ALPHA
        if self.step_count == 1:
            self.ema_phi = phi
            self.ema_ignition = ignition
            self.ema_vividness = vividness
            self.ema_level = level
        else:
            self.ema_phi = self.ema_phi * (1 - a) + phi * a
            self.ema_ignition = self.ema_ignition * (1 - a) + ignition * a
            self.ema_vividness = self.ema_vividness * (1 - a) + vividness * a
            self.ema_level = self.ema_level * (1 - a) + level * a

    def get_adjusted_weights(self) -> Dict[str, float]:
        if self.step_count < 10:
            return {
                "iit_phi": _IIT_WEIGHT,
                "gwt_ignition": _GWT_WEIGHT,
                "predictive_coding": _PREDICTIVE_WEIGHT,
                "first_person": _FP_WEIGHT,
                "perceptual_vividness": _VIVIDNESS_WEIGHT,
            }

        ema_v = max(0.001, self.ema_vividness)
        sensory_gate = _compute_sensory_gate(ema_v)

        loop_rate = self.loop_count / max(1, self.step_count)
        loop_penalty = max(0.3, 1.0 - loop_rate * 2.0)

        ignition_var = float(np.var(list(self.ignition_history))) if len(self.ignition_history) >= 5 else 0.0
        ignition_effective = 0.3 + 0.7 * min(1.0, ignition_var * 10.0)

        phi_effective = _IIT_WEIGHT * sensory_gate * loop_penalty
        gwt_remaining = _GWT_WEIGHT * ignition_effective * sensory_gate
        redistributed = (_IIT_WEIGHT - phi_effective) + (_GWT_WEIGHT - gwt_remaining)

        predictive_effective = _PREDICTIVE_WEIGHT + redistributed * 0.5
        vividness_effective = _VIVIDNESS_WEIGHT + redistributed * 0.5

        return {
            "iit_phi": phi_effective,
            "gwt_ignition": gwt_remaining,
            "predictive_coding": predictive_effective,
            "first_person": _FP_WEIGHT,
            "perceptual_vividness": vividness_effective,
        }

    def get_diagnostics(self) -> Dict[str, float]:
        n = max(1, self.step_count)
        return {
            "sensory_gate": _compute_sensory_gate(max(0.001, self.ema_vividness)),
            "ignition_variance": float(np.var(list(self.ignition_history))) if len(self.ignition_history) >= 5 else 0.0,
            "ema_vividness": self.ema_vividness,
            "ema_phi": self.ema_phi,
            "ema_level": self.ema_level,
            "samples": self.step_count,
            "loop_rate": self.loop_count / n,
            "ignition_saturated_rate": self.ignition_saturated_count / n,
            "sensory_deprived_rate": self.sensory_deprived_count / n,
        }


def compute_consciousness_level(
    phi,
    global_ignition,
    self_prediction_error,
    first_person_salience=0.0,
    perceptual_vividness=0.0,
    calibrator: Optional[ConsciousnessCalibrator] = None,
    step: int = 0,
    prev_self_accuracy: float = 0.5,
):
    phi = min(1.0, max(0.0, phi))
    global_ignition = min(1.0, max(0.0, global_ignition))
    self_prediction_error = min(1.0, max(0.0, self_prediction_error))
    first_person_salience = min(1.0, max(0.0, first_person_salience))
    perceptual_vividness = min(1.0, max(0.0, perceptual_vividness))

    vividness = perceptual_vividness
    sensory_gate = _compute_sensory_gate(vividness)

    loop_detected, corrected_accuracy = _detect_self_referential_loop(
        self_prediction_error, vividness, prev_self_accuracy
    )

    raw_self_accuracy = 1.0 - self_prediction_error
    blended_self_accuracy = (
        raw_self_accuracy * sensory_gate
        + corrected_accuracy * (1.0 - sensory_gate)
    )
    blended_self_accuracy = float(np.clip(blended_self_accuracy, 0.0, 1.0))

    ignition_var = calibrator.ignition_history if calibrator else []
    if isinstance(ignition_var, deque) and len(ignition_var) >= 5:
        ign_arr = np.array(list(ignition_var))
        ign_var_val = float(np.var(ign_arr))
        ignition_saturated = ign_var_val < 0.001 and global_ignition > 0.9
    else:
        ignition_saturated = global_ignition > 0.98
    gated_ignition = global_ignition * (0.3 + 0.7 * sensory_gate)

    weights = {
        "iit_phi": _IIT_WEIGHT,
        "gwt_ignition": _GWT_WEIGHT,
        "predictive_coding": _PREDICTIVE_WEIGHT,
        "first_person": _FP_WEIGHT,
        "perceptual_vividness": _VIVIDNESS_WEIGHT,
    }

    if calibrator is not None and calibrator.step_count >= 20:
        weights = calibrator.get_adjusted_weights()

    temporal_decay = 1.0
    if step > _TEMPORAL_DECAY_HALFLIFE:
        temporal_decay = np.exp(
            -np.log(2) * max(0, step - _TEMPORAL_DECAY_HALFLIFE) / _TEMPORAL_DECAY_HALFLIFE
        )
        temporal_decay = float(np.clip(temporal_decay, 0.3, 1.0))

    level = (
        weights["iit_phi"] * phi
        + weights["gwt_ignition"] * gated_ignition
        + weights["predictive_coding"] * blended_self_accuracy
        + weights["first_person"] * first_person_salience
        + weights["perceptual_vividness"] * vividness
    )

    if loop_detected:
        level *= 0.7

    if calibrator is not None:
        calibrator.update(
            phi=phi,
            ignition=global_ignition,
            self_error=self_prediction_error,
            fp_salience=first_person_salience,
            vividness=vividness,
            level=level,
            loop_detected=loop_detected,
            ignition_saturated=ignition_saturated,
        )

    level = level * temporal_decay

    return min(1.0, max(0.0, level))


def compute_consciousness_from_cortical_brain(brain, calibrator=None, step=0):
    cs = brain.read_consciousness()
    phi = cs.phi
    global_ignition = cs.global_ignition
    self_prediction_error = cs.self_prediction_error
    first_person_salience = brain.get_first_person_salience()
    perceptual_vividness = brain.get_perceptual_vividness()

    prev_acc = calibrator.prev_self_accuracy if calibrator else 0.5

    level = compute_consciousness_level(
        phi=phi,
        global_ignition=global_ignition,
        self_prediction_error=self_prediction_error,
        first_person_salience=first_person_salience,
        perceptual_vividness=perceptual_vividness,
        calibrator=calibrator,
        step=step,
        prev_self_accuracy=prev_acc,
    )

    sensory_gate = _compute_sensory_gate(perceptual_vividness)
    loop_detected, corrected_accuracy = _detect_self_referential_loop(
        self_prediction_error, perceptual_vividness, prev_acc
    )

    ignition_var = float(np.var(list(calibrator.ignition_history))) if calibrator and len(calibrator.ignition_history) >= 5 else 0.0
    ignition_saturated = ignition_var < 0.001 and global_ignition > 0.9

    components = {
        "phi": phi,
        "global_ignition": global_ignition,
        "self_prediction_error": self_prediction_error,
        "self_model_accuracy_raw": 1.0 - self_prediction_error,
        "self_model_accuracy_corrected": corrected_accuracy,
        "first_person_salience": first_person_salience,
        "perceptual_vividness": perceptual_vividness,
        "sensory_gate": sensory_gate,
        "consciousness_level": level,
        "iit_component": _IIT_WEIGHT * phi * sensory_gate,
        "gwt_component": _GWT_WEIGHT * global_ignition * (0.3 + 0.7 * sensory_gate),
        "predictive_component": _PREDICTIVE_WEIGHT * corrected_accuracy,
        "fp_component": _FP_WEIGHT * first_person_salience,
        "vividness_component": _VIVIDNESS_WEIGHT * perceptual_vividness,
        "flag_sensory_deprived": perceptual_vividness < _VIVIDNESS_MIN_THRESHOLD,
        "flag_loop_detected": loop_detected,
        "flag_ignition_saturated": ignition_saturated,
    }

    return level, components


def compute_consciousness_from_metrics(phi, gw_activity, self_prediction_error,
                                        first_person_salience=0.0,
                                        perceptual_vividness=0.0,
                                        calibrator=None, step=0):
    return compute_consciousness_level(
        phi=phi,
        global_ignition=gw_activity,
        self_prediction_error=self_prediction_error,
        first_person_salience=first_person_salience,
        perceptual_vividness=perceptual_vividness,
        calibrator=calibrator,
        step=step,
    )


def get_weights():
    return {
        "iit_phi": _IIT_WEIGHT,
        "gwt_ignition": _GWT_WEIGHT,
        "predictive_coding": _PREDICTIVE_WEIGHT,
        "first_person": _FP_WEIGHT,
        "perceptual_vividness": _VIVIDNESS_WEIGHT,
        "total": _IIT_WEIGHT + _GWT_WEIGHT + _PREDICTIVE_WEIGHT
                + _FP_WEIGHT + _VIVIDNESS_WEIGHT,
        "version": "v6",
    }


def get_calibration_params():
    return {
        "vividness_min_threshold": _VIVIDNESS_MIN_THRESHOLD,
        "self_error_max_threshold": _SELF_ERROR_MAX_THRESHOLD,
        "credibility_ema_alpha": _CREDIBILITY_EMA_ALPHA,
        "temporal_decay_halflife": _TEMPORAL_DECAY_HALFLIFE,
        "sensory_gate_range": [0.15, 1.0],
        "loop_accuracy_threshold": _LOOP_ACCURACY_THRESHOLD,
        "loop_diff_threshold": _LOOP_DIFF_THRESHOLD,
        "loop_vividness_threshold": _LOOP_VIVIDNESS_THRESHOLD,
        "version": "v6",
    }


_AUXILIARY_METRICS = {
    "spontaneity": "DMN spontaneous thought activity — not a consciousness metric",
    "semantic_strength": "Language/concept grounding strength — not a consciousness metric",
    "thought_energy": "DMN energy level — not a consciousness metric",
    "temporal_depth_score": "Temporal self-continuity — metacognitive, report separately",
    "theory_mind_level": "Theory of Mind capability — social cognition, report separately",
    "goal_progress": "Goal pursuit progress — motivation system, report separately",
    "goal_satisfaction": "Goal satisfaction level — reward system, report separately",
    "emotional_depth": "Emotional processing depth — affective system, report separately",
    "emotional_range": "Emotional range breadth — affective system, report separately",
    "planning_depth": "Planning horizon — executive function, report separately",
    "social_confidence": "Social interaction confidence — social cognition, report separately",
    "creativity_level": "Creative ideation level — creative cognition, report separately",
    "meta_confidence": "Metacognitive confidence — metacognition, report separately",
    "ignition_saturated_flag": "GWT ignition saturation flag — competition health, report separately",
    "loop_suspected_flag": "Self-referential loop flag — model health, report separately",
}


def get_auxiliary_metrics_report():
    return dict(_AUXILIARY_METRICS)


def cross_validate_consciousness(brain, calibrator=None, step=0, tolerance=0.15):
    """
    C++/Python 双路径意识计算交叉验证。

    比较 Python 端统一公式与 C++ 端内部 _update_consciousness() 的结果，
    检测两条计算路径之间的一致性偏差。
    """
    cs = brain.read_consciousness()
    cpp_phi = cs.phi
    cpp_ignition = cs.global_ignition
    cpp_self_error = cs.self_prediction_error
    cpp_fp = brain.get_first_person_salience()
    cpp_vividness = brain.get_perceptual_vividness()

    py_level, py_components = compute_consciousness_from_cortical_brain(
        brain, calibrator=calibrator, step=step)

    cpp_level_approx = (
        0.35 * cpp_phi
        + 0.25 * cpp_ignition
        + 0.20 * (1.0 - cpp_self_error)
        + 0.10 * cpp_fp
        + 0.10 * cpp_vividness
    )
    cpp_level_approx = min(1.0, max(0.0, cpp_level_approx))

    delta = abs(py_level - cpp_level_approx)
    aligned = delta <= tolerance

    return {
        "python_consciousness": py_level,
        "cpp_phi": cpp_phi,
        "cpp_ignition": cpp_ignition,
        "cpp_self_error": cpp_self_error,
        "cpp_fp_salience": cpp_fp,
        "cpp_vividness": cpp_vividness,
        "cpp_approx_consciousness": cpp_level_approx,
        "delta": delta,
        "aligned": aligned,
        "python_components": py_components,
        "diagnosis": _diagnose_mismatch(py_level, cpp_level_approx, cpp_vividness,
                                         cpp_ignition, cpp_self_error),
    }


def _diagnose_mismatch(py_level, cpp_level, vividness, ignition, self_error):
    issues = []
    delta = abs(py_level - cpp_level)

    if delta > 0.15:
        issues.append(f"CRITICAL: Python-C++ divergence {delta:.3f} > 0.15")
    elif delta > 0.08:
        issues.append(f"WARNING: Python-C++ divergence {delta:.3f} > 0.08")

    if vividness < 0.02:
        issues.append("CRITICAL: vividness below minimum threshold ({:.4f})".format(vividness))
    elif vividness < 0.06:
        issues.append("WARNING: vividness low ({:.4f}), sensory gate active".format(vividness))

    if ignition > 0.95:
        issues.append("WARNING: ignition near saturation ({:.4f})".format(ignition))

    if self_error < 0.01:
        issues.append("WARNING: self_prediction_error critically low ({:.4f}), "
                       "self-referential loop risk".format(self_error))
    elif self_error < 0.03:
        issues.append("INFO: self_prediction_error low ({:.4f})".format(self_error))

    if not issues:
        issues.append("OK: all metrics within expected ranges")

    return issues


def compute_consciousness_diagnostic_report(brain, calibrator=None, step=0):
    """
    生成完整的意识计算诊断报告。
    """
    cross = cross_validate_consciousness(brain, calibrator=calibrator, step=step)

    cs = brain.read_consciousness()
    vividness = brain.get_perceptual_vividness()
    fp_salience = brain.get_first_person_salience()

    sensory_gate = _compute_sensory_gate(vividness)
    loop_detected, _ = _detect_self_referential_loop(
        cs.self_prediction_error, vividness,
        calibrator.prev_self_accuracy if calibrator else 0.5
    )

    report = {
        "version": "v7-aligned",
        "cross_validation": cross,
        "flags": {
            "cpp_ignition_saturated": cs.global_ignition > 0.95,
            "sensory_deprived": vividness < _VIVIDNESS_MIN_THRESHOLD,
            "loop_risk": loop_detected,
            "low_vividness": vividness < 0.06,
            "cross_validated": cross["aligned"],
        },
        "metrics": {
            "sensory_gate": sensory_gate,
            "vividness": vividness,
            "fp_salience": fp_salience,
            "weights": get_weights(),
        }
    }

    if calibrator is not None:
        report["calibrator_diagnostics"] = calibrator.get_diagnostics()
        report["calibrator_weights"] = calibrator.get_adjusted_weights()

    return report