import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from cognitive.unified_consciousness import (
    compute_consciousness_level,
    compute_consciousness_from_metrics,
    get_weights,
    get_auxiliary_metrics_report,
    get_calibration_params,
    ConsciousnessCalibrator,
    _compute_sensory_gate,
    _detect_self_referential_loop,
    compute_consciousness_diagnostic_report,
    _diagnose_mismatch,
)


class TestUnifiedConsciousnessFormula:
    """验证统一意识公式的正确性。"""

    def test_weights_sum_to_one(self):
        weights = get_weights()
        total = weights["total"]
        assert abs(total - 1.0) < 1e-9, f"权重和应为 1.0, 实际: {total}"
        assert weights["iit_phi"] == 0.35
        assert weights["gwt_ignition"] == 0.25
        assert weights["predictive_coding"] == 0.20
        assert weights["first_person"] == 0.10
        assert weights["perceptual_vividness"] == 0.10

    def test_all_zeros_produces_zero(self):
        level = compute_consciousness_level(
            phi=0.0,
            global_ignition=0.0,
            self_prediction_error=1.0,
            first_person_salience=0.0,
            perceptual_vividness=0.0,
        )
        assert level == 0.0, f"全零输入应产生 0.0, 实际: {level}"

    def test_all_ones_produces_one(self):
        level = compute_consciousness_level(
            phi=1.0,
            global_ignition=1.0,
            self_prediction_error=0.0,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert abs(level - 1.0) < 1e-9, f"全最佳输入应产生 1.0, 实际: {level}"

    def test_bounded_between_zero_and_one(self):
        import random
        random.seed(42)
        for _ in range(1000):
            phi = random.random()
            ignition = random.random()
            error = random.random()
            fp = random.random()
            vivid = random.random()
            level = compute_consciousness_level(
                phi=phi,
                global_ignition=ignition,
                self_prediction_error=error,
                first_person_salience=fp,
                perceptual_vividness=vivid,
            )
            assert 0.0 <= level <= 1.0, f"超出[0,1]范围: {level}"

    def test_clamping_extreme_values(self):
        level = compute_consciousness_level(
            phi=5.0,
            global_ignition=-0.5,
            self_prediction_error=2.0,
            first_person_salience=3.0,
            perceptual_vividness=-1.0,
        )
        assert 0.0 <= level <= 1.0, f"极端值应被截断: {level}"

    def test_iit_weight_dominates(self):
        high_phi = compute_consciousness_level(
            phi=0.9, global_ignition=0.2,
            self_prediction_error=0.8,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        low_phi = compute_consciousness_level(
            phi=0.1, global_ignition=0.2,
            self_prediction_error=0.8,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert high_phi > low_phi, "高 phi 应产生更高意识水平"

    def test_self_error_inverse(self):
        low_error = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.1,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        high_error = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.9,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert low_error > high_error, "低预测误差应产生更高意识水平"

    def test_from_metrics_wrapper(self):
        level = compute_consciousness_from_metrics(
            phi=0.5, gw_activity=0.5,
            self_prediction_error=0.5,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        expected = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.5,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert abs(level - expected) < 1e-9

    def test_mathematical_consistency(self):
        component_sum = (
            0.35 * 0.5 + 0.25 * 0.5 + 0.20 * 0.5 + 0.10 * 1.0 + 0.10 * 1.0
        )
        level = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.5,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert abs(level - component_sum) < 1e-9


class TestV5Features:
    """验证 v5 新增功能。"""

    def test_sensory_gate_range(self):
        gate_0 = _compute_sensory_gate(0.0)
        gate_005 = _compute_sensory_gate(0.05)
        gate_03 = _compute_sensory_gate(0.3)
        gate_1 = _compute_sensory_gate(1.0)

        assert 0.14 <= gate_0 <= 0.16, f"vividness=0 → gate≈0.15, 实际: {gate_0}"
        assert 0.25 <= gate_005 <= 0.35, f"vividness=0.05 → gate≈0.28, 实际: {gate_005}"
        assert gate_03 > gate_005, "gate 应随 vividness 单调递增"
        assert gate_1 > gate_03
        assert abs(gate_1 - 1.0) < 1e-9, f"vividness=1 → gate=1.0, 实际: {gate_1}"

    def test_anti_loop_correction_detects_loop(self):
        loop, corr = _detect_self_referential_loop(
            self_prediction_error=0.005,
            perceptual_vividness=0.03,
            prev_self_accuracy=0.996,
        )
        assert loop, "严重自指循环应被检测到"
        assert corr < 0.7, f"严重自指循环应得到 < 0.7 的修正, 实际: {corr}"

    def test_anti_loop_correction_normal(self):
        loop, corr = _detect_self_referential_loop(
            self_prediction_error=0.1,
            perceptual_vividness=0.5,
            prev_self_accuracy=0.9,
        )
        assert not loop, "正常情况不应检测到循环"
        assert abs(corr - 0.9) < 0.1, f"正常情况应接近原始 accuracy, 实际: {corr}"

    def test_vividness_gates_consciousness(self):
        high_vivid = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.01,
            first_person_salience=0.5,
            perceptual_vividness=0.8,
        )
        low_vivid = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.01,
            first_person_salience=0.5,
            perceptual_vividness=0.05,
        )
        assert high_vivid > low_vivid, "高 vividness 应产生更高意识水平"

    def test_calibration_params_defined(self):
        params = get_calibration_params()
        assert 'vividness_min_threshold' in params
        assert 'temporal_decay_halflife' in params
        assert params['vividness_min_threshold'] == 0.02

    def test_temporal_decay_applies(self):
        level_early = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.5,
            first_person_salience=0.5,
            perceptual_vividness=1.0,
            step=10,
        )
        level_late = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.5,
            first_person_salience=0.5,
            perceptual_vividness=1.0,
            step=1000,
        )
        assert level_late <= level_early, "后期应有时序衰减"

    def test_calibrator_tracks_metrics(self):
        cal = ConsciousnessCalibrator(window_size=50)
        for i in range(30):
            compute_consciousness_level(
                phi=0.3 + i * 0.01,
                global_ignition=0.5,
                self_prediction_error=0.1,
                first_person_salience=0.2,
                perceptual_vividness=0.1,
                calibrator=cal,
                step=i,
            )
        assert cal.step_count == 30
        assert 0.3 < cal.ema_phi < 0.7
        diag = cal.get_diagnostics()
        assert diag['samples'] == 30
        assert 'sensory_gate' in diag

    def test_calibrator_adjusts_weights(self):
        cal = ConsciousnessCalibrator(window_size=50)
        for i in range(25):
            compute_consciousness_level(
                phi=0.5, global_ignition=1.0,
                self_prediction_error=0.01,
                first_person_salience=0.1,
                perceptual_vividness=0.05,
                calibrator=cal,
                step=i,
            )
        weights = cal.get_adjusted_weights()
        assert weights['iit_phi'] < 0.35, "低 vividness 应衰减 IIT 权重"
        assert weights['gwt_ignition'] < 0.25, "低 vividness 应衰减 GWT 权重"


class TestAuxiliaryMetricsExclusion:
    """验证意识无关指标已被正确排除。"""

    def test_auxiliary_metrics_defined(self):
        report = get_auxiliary_metrics_report()
        assert len(report) > 0
        assert "social_confidence" in report
        assert "creativity_level" in report
        assert "goal_progress" in report

    def test_no_auxiliary_metric_in_formula(self):
        weights = get_weights()
        for key in get_auxiliary_metrics_report():
            assert key not in weights, f"{key} 不应出现在意识公式权重中"


class TestTheoreticalGrounding:
    """验证公式符合 IIT/GWT/PC 理论约束。"""

    def test_phi_monotonic(self):
        base = 0.3
        results = []
        for phi in np.linspace(0, 1, 11):
            level = compute_consciousness_level(
                phi=phi, global_ignition=base,
                self_prediction_error=1.0 - base,
                first_person_salience=1.0,
                perceptual_vividness=1.0,
            )
            results.append(level)
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1] - 1e-9, \
                f"phi 单调性违反: {results[i-1]} > {results[i]}"

    def test_ignition_monotonic(self):
        base = 0.3
        results = []
        for ignition in np.linspace(0, 1, 11):
            level = compute_consciousness_level(
                phi=base, global_ignition=ignition,
                self_prediction_error=1.0 - base,
                first_person_salience=1.0,
                perceptual_vividness=1.0,
            )
            results.append(level)
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1] - 1e-9, \
                f"ignition 单调性违反"


class TestConsciousnessMetricsIntegration:
    """验证 ConsciousnessMetrics 类与统一公式的集成。"""

    def test_metrics_uses_unified_formula(self):
        from cognitive.consciousness_metrics import ConsciousnessMetrics

        metrics = ConsciousnessMetrics(gw_population_size=200)
        metrics.update_phi(0.5)
        metrics.update_self_error(0.2)
        metrics.update_qualia(
            first_person_salience=0.3,
            perceptual_vividness=0.4,
        )

        level = metrics.compute_consciousness_level(gw_activity=0.6)

        direct = compute_consciousness_level(
            phi=0.5, global_ignition=0.6,
            self_prediction_error=0.2,
            first_person_salience=0.3,
            perceptual_vividness=0.4,
        )

        assert abs(level - direct) < 1e-9, \
            f"Metrics 应与直接调用一致: {level} vs {direct}"

    def test_metrics_backward_compat(self):
        from cognitive.consciousness_metrics import ConsciousnessMetrics

        metrics = ConsciousnessMetrics()
        level = metrics.compute_consciousness_level(
            gw_activity=0.5, self_model=None
        )
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0


class TestCrossValidation:
    """验证 v7 新增的跨路径验证和诊断功能。"""

    def test_diagnose_mismatch_all_ok(self):
        issues = _diagnose_mismatch(0.5, 0.52, 0.3, 0.6, 0.1)
        assert len(issues) == 1
        assert "OK" in issues[0]

    def test_diagnose_mismatch_critical_divergence(self):
        issues = _diagnose_mismatch(0.3, 0.55, 0.3, 0.6, 0.1)
        assert any("CRITICAL" in i for i in issues)

    def test_diagnose_mismatch_warning_divergence(self):
        issues = _diagnose_mismatch(0.5, 0.41, 0.3, 0.6, 0.1)
        assert any("WARNING" in i for i in issues) or any("OK" in i for i in issues)

    def test_diagnose_warns_low_vividness(self):
        issues = _diagnose_mismatch(0.5, 0.5, 0.01, 0.6, 0.1)
        assert any("vividness" in i.lower() for i in issues)

    def test_diagnose_warns_saturated_ignition(self):
        issues = _diagnose_mismatch(0.5, 0.5, 0.3, 0.98, 0.1)
        assert any("saturation" in i.lower() for i in issues)

    def test_diagnose_warns_low_self_error(self):
        issues = _diagnose_mismatch(0.5, 0.5, 0.3, 0.6, 0.005)
        assert any("self" in i.lower() for i in issues)


class TestV7Enhancements:
    """验证 v7 版本增强功能。"""

    def test_self_error_never_zero(self):
        level = compute_consciousness_level(
            phi=0.5, global_ignition=0.5,
            self_prediction_error=0.0,
            first_person_salience=1.0,
            perceptual_vividness=1.0,
        )
        assert level > 0.0

    def test_anti_saturation_sensory_gate(self):
        gate_low = _compute_sensory_gate(0.02)
        gate_high = _compute_sensory_gate(0.5)
        assert gate_low < 0.65, f"低 vividness 应限制 gate: {gate_low}"
        assert gate_high > 0.6, f"高 vividness 应释放 gate: {gate_high}"

    def test_loop_protection_robust(self):
        loop, corr = _detect_self_referential_loop(
            self_prediction_error=0.002,
            perceptual_vividness=0.01,
            prev_self_accuracy=0.999,
        )
        assert loop, "极端条件应检测到自指循环"
        assert corr < 0.7

        loop2, corr2 = _detect_self_referential_loop(
            self_prediction_error=0.1,
            perceptual_vividness=0.3,
            prev_self_accuracy=0.9,
        )
        assert not loop2, "正常条件不应误报"

    def test_version_number(self):
        weights = get_weights()
        assert weights["version"] == "v6"
        total = weights["total"]
        assert abs(total - 1.0) < 1e-9


class TestIntegrationStress:
    """压力测试和边界条件。"""

    def test_rapid_oscillation(self):
        results = []
        for i in range(100):
            level = compute_consciousness_level(
                phi=float(i % 2),
                global_ignition=float((i + 1) % 2),
                self_prediction_error=float((i * 7) % 10) / 10.0,
                first_person_salience=float(i % 3) / 3.0,
                perceptual_vividness=float((i % 5) + 1) / 6.0,
            )
            results.append(level)
            assert 0.0 <= level <= 1.0, f"step {i}: {level} 越界"
        assert min(results) < max(results), "结果应有变化"

    def test_long_run_stability(self):
        cal = ConsciousnessCalibrator(window_size=200)
        levels = []
        for step in range(500):
            level = compute_consciousness_level(
                phi=0.3 + 0.4 * (float(step % 50) / 50.0),
                global_ignition=0.3 + 0.3 * (float((step % 30)) / 30.0),
                self_prediction_error=0.05 + 0.15 * (float(step % 20) / 20.0),
                first_person_salience=0.1 + 0.2 * (float(step % 10) / 10.0),
                perceptual_vividness=0.05 + 0.3 * (float((step % 25)) / 25.0),
                calibrator=cal,
                step=step,
            )
            levels.append(level)
            assert 0.0 <= level <= 1.0
        avg = np.mean(levels)
        assert 0.1 < avg < 0.9, f"长期运行平均值应合理: {avg}"
        assert cal.step_count == 500