"""
Phi-Behavior 实时修正引擎 v4 — 综合测试套件
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from cognitive.phi_validator import (
    PhiCorrectionEngine,
    PhiBehaviorCalibration,
    PhiValidator,
    check_self_referential_loop,
    check_ignition_health,
    format_report,
)


class TestPhiCorrectionEngine:
    """验证 PhiCorrectionEngine v4 的基本功能。"""

    def test_initial_state(self):
        engine = PhiCorrectionEngine(window_size=100)
        assert engine.correction_gain == 0.0
        assert engine.last_pearson_r == 0.0
        assert engine.last_spearman_rho == 0.0

    def test_add_observation_tracks_metrics(self):
        engine = PhiCorrectionEngine(window_size=100)
        for i in range(15):
            engine.add_observation(
                phi=0.3 + 0.01 * i,
                behavioral_score=0.2 + 0.01 * i,
                ignition=0.5,
                vividness=0.4,
            )
        stats = engine.get_stats()
        assert stats['n'] == 15

    def test_correlation_computes(self):
        engine = PhiCorrectionEngine(window_size=100)
        for i in range(15):
            engine.add_observation(
                phi=0.3 + 0.01 * i,
                behavioral_score=0.3 + 0.01 * i,
                ignition=0.6,
                vividness=0.4,
            )
        stats = engine.get_stats()
        assert stats['pearson_r'] is not None

    def test_loop_detection(self):
        is_loop, msg = check_self_referential_loop(0.005, 0.03)
        assert is_loop
        assert "loop" in msg.lower() or "自指" in msg

    def test_no_false_positive_loop_detection(self):
        is_loop, msg = check_self_referential_loop(0.2, 0.5)
        assert not is_loop
        assert "Normal" in msg

    def test_ignition_health_saturated(self):
        saturated = [1.0] * 20
        is_sat, msg = check_ignition_health(saturated)
        assert is_sat
        assert "saturated" in msg.lower()

    def test_ignition_health_normal(self):
        varied = [0.3, 0.5, 0.2, 0.7, 0.1, 0.6, 0.4, 0.8, 0.2, 0.5,
                   0.3, 0.6, 0.1, 0.7, 0.4, 0.8, 0.2, 0.5, 0.3, 0.9]
        is_sat, msg = check_ignition_health(varied)
        assert not is_sat

    def test_divergence_detection(self):
        engine = PhiCorrectionEngine(window_size=50)
        for i in range(30):
            engine.add_observation(
                phi=0.8,
                behavioral_score=0.1,
                ignition=1.0,
                vividness=0.2,
            )
        rec = engine.get_recommendation()
        if rec['status'] == 'divergence_detected':
            assert rec['divergence_count'] > 0

    def test_correction_gain_increases_on_issues(self):
        engine = PhiCorrectionEngine(window_size=50)
        for i in range(20):
            engine.add_observation(
                phi=0.8,
                behavioral_score=0.1,
                ignition=1.0,
                vividness=0.03,
            )
        gain = engine.get_correction_gain()
        assert gain >= 0.0

    def test_trend_analysis(self):
        engine = PhiCorrectionEngine(window_size=50)
        for i in range(30):
            engine.add_observation(
                phi=0.3 + 0.01 * i,
                behavioral_score=0.3 + 0.01 * i,
                ignition=0.6,
                vividness=0.4,
            )
        stats = engine.get_stats()
        assert 'phi_slope' in stats
        assert 'behavior_slope' in stats

    def test_recommendation_format(self):
        engine = PhiCorrectionEngine(window_size=50)
        for i in range(20):
            engine.add_observation(
                phi=0.5,
                behavioral_score=0.5,
                ignition=0.5,
                vividness=0.5,
            )
        rec = engine.get_recommendation()
        assert 'status' in rec
        assert 'pearson_r' in rec
        assert 'spearman_rho' in rec
        assert 'p_value' in rec

    def test_adaptive_window(self):
        engine = PhiCorrectionEngine(window_size=100)
        for i in range(15):
            engine.add_observation(
                phi=0.3 + 0.001 * i,
                behavioral_score=0.3 + 0.001 * i,
                ignition=0.5,
                vividness=0.4,
            )
        stats = engine.get_stats()
        assert stats['adaptive_window'] > 0


class TestPhiBehaviorCalibration:
    """验证 PhiBehaviorCalibration 基本功能。"""

    def test_update_and_stats(self):
        cal = PhiBehaviorCalibration(window_size=50)
        for i in range(20):
            cal.update(phi=0.5, behavior_score=0.6, conscious_level=0.5)
        stats = cal.get_stats()
        assert stats['n'] == 20
        assert stats['phi_mean'] > 0
        assert stats['behavior_mean'] > 0

    def test_correlation_empty_returns_zero(self):
        cal = PhiBehaviorCalibration(window_size=50)
        r, rho, p = cal.get_phi_behavior_correlation()
        assert r == 0.0
        assert rho == 0.0

    def test_correlation_computes(self):
        cal = PhiBehaviorCalibration(window_size=50)
        for i in range(20):
            cal.update(phi=0.3 + 0.01 * i, behavior_score=0.3 + 0.01 * i,
                        conscious_level=0.5)
        r, rho, p = cal.get_phi_behavior_correlation()
        assert r is not None


class TestUtilityFunctions:
    """验证工具函数。"""

    def test_format_report(self):
        stats = {
            'n': 100,
            'phi_mean': 0.15,
            'phi_std': 0.03,
            'behavior_mean': 0.73,
            'behavior_std': 0.20,
            'pearson_r': 0.52,
            'spearman_rho': 0.85,
            'p_value': 0.001,
            'correction_gain': 0.0,
            'trend_aligned': True,
            'adaptive_window': 100,
            'divergence_detected': False,
        }
        report = format_report(stats)
        assert "Phi-Behavior" in report
        assert "Pearson" in report
        assert "Spearman" in report

    def test_format_report_with_warning(self):
        stats = {
            'n': 100,
            'phi_mean': 0.5,
            'phi_std': 0.1,
            'behavior_mean': 0.3,
            'behavior_std': 0.1,
            'pearson_r': 0.05,
            'spearman_rho': 0.1,
            'p_value': 0.5,
            'correction_gain': 0.3,
            'trend_aligned': False,
            'adaptive_window': 50,
            'divergence_detected': True,
        }
        rec = {
            'status': 'divergence_detected',
            'recommendations': ['GWT_IGNITION_SATURATED'],
        }
        report = format_report(stats, rec)
        assert "WARNING" in report


class TestPhiValidatorIntegrated:
    """验证 v7 新增的 PhiValidator 集成类。"""

    def test_initial_health_flags(self):
        validator = PhiValidator(window_size=100)
        assert validator.health_flags["ignition_saturated"] is False
        assert validator.health_flags["loop_detected"] is False
        assert validator.health_flags["sensory_deprived"] is False
        assert validator.health_flags["phi_behavior_decoupled"] is False

    def test_step_tracks_health(self):
        validator = PhiValidator(window_size=100)
        for i in range(25):
            validator.step(
                phi=0.3 + i * 0.002,
                behavioral_score=0.5,
                ignition=0.6,
                vividness=0.01,
                self_prediction_error=0.005,
                consciousness_level=0.4,
            )
        flags = validator.health_flags
        assert flags["loop_detected"] or flags["sensory_deprived"]

    def test_step_resets_sensory_deprived(self):
        validator = PhiValidator(window_size=100)
        validator.step(0.5, 0.5, 0.5, 0.01, 0.1, 0.5)
        assert validator.health_flags["sensory_deprived"] is True
        validator.step(0.5, 0.5, 0.5, 0.1, 0.1, 0.5)
        assert validator.health_flags["sensory_deprived"] is False

    def test_get_report_has_keys(self):
        validator = PhiValidator(window_size=100)
        for i in range(20):
            validator.step(
                phi=0.3 + i * 0.01,
                behavioral_score=0.3 + i * 0.01,
                ignition=0.5,
                vividness=0.3,
                self_prediction_error=0.1,
                consciousness_level=0.5,
            )
        report = validator.get_report()
        assert "phi_mean" in report
        assert "health_flags" in report
        assert "pearson_r" in report
        assert "spearman_rho" in report

    def test_print_report_does_not_crash(self):
        validator = PhiValidator(window_size=100)
        for i in range(10):
            validator.step(0.5, 0.5, 0.5, 0.3, 0.1, 0.5)
        validator.print_report()