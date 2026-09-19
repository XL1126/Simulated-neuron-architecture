import numpy as np
import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from cognitive.unified_consciousness import (
    compute_consciousness_level,
    compute_consciousness_from_cortical_brain,
)


class AblationStudy:
    """
    SNA 模块摘除对照实验框架。

    通过逐一关闭认知模块来测量每个模块对意识水平的因果贡献。
    每个模块被"摘除"时，其输出被替换为零向量或默认值，
    然后测量意识水平的变化。

    用法:
        study = AblationStudy()
        study.register_brain_factory(create_brain)
        study.add_ablation("visual", disable_visual)
        study.add_ablation("hippocampus", disable_hippocampus)
        results = study.run_all(n_steps=200)
        study.print_report(results)
    """

    def __init__(self):
        self.brain_factory = None
        self.ablations = {}
        self.control_consciousness = None
        self.control_components = None

    def register_brain_factory(self, factory):
        """
        注册大脑工厂函数。factory() 应返回一个已初始化的 brain 对象。
        """
        self.brain_factory = factory

    def add_ablation(self, name, disable_fn):
        """
        添加一个摘除条件。

        参数:
            name: 摘除模块名称
            disable_fn: fn(brain) -> brain, 返回被摘除后的 brain
        """
        self.ablations[name] = disable_fn

    def _run_control(self, n_steps=200):
        """
        运行对照组 (所有模块正常工作)。
        """
        brain = self.brain_factory()
        for _ in range(n_steps):
            brain.step(0.0)
        try:
            level, comps = compute_consciousness_from_cortical_brain(brain)
            return level, comps
        except Exception:
            cs = brain.read_consciousness()
            return cs.phi, {"phi": cs.phi, "global_ignition": cs.global_ignition}

    def _run_ablation(self, name, disable_fn, n_steps=200):
        """
        运行单个摘除实验。
        """
        brain = self.brain_factory()
        brain = disable_fn(brain)
        for _ in range(n_steps):
            brain.step(0.0)
        try:
            level, comps = compute_consciousness_from_cortical_brain(brain)
            return level, comps
        except Exception:
            cs = brain.read_consciousness()
            return cs.phi, {"phi": cs.phi, "global_ignition": cs.global_ignition}

    def run_all(self, n_steps=200):
        """
        运行完整的摘除实验套件。

        返回:
            dict: {
                'control': {'consciousness': float, 'components': dict},
                'ablations': {name: {'consciousness': float, 'delta': float, ...}}
            }
        """
        if self.brain_factory is None:
            raise ValueError("请先调用 register_brain_factory()")

        results = {}

        self.control_consciousness, self.control_components = \
            self._run_control(n_steps)
        results['control'] = {
            'consciousness': self.control_consciousness,
            'components': self.control_components,
        }

        results['ablations'] = {}
        for name, disable_fn in self.ablations.items():
            level, comps = self._run_ablation(name, disable_fn, n_steps)
            delta = self.control_consciousness - level
            relative_contribution = delta / max(1e-6, self.control_consciousness)

            results['ablations'][name] = {
                'consciousness': level,
                'delta': delta,
                'relative_contribution': relative_contribution,
                'components': comps,
            }

        return results

    def print_report(self, results):
        """
        打印格式化的摘除实验报告。
        """
        print("=" * 70)
        print("SNA Ablation Study Report")
        print("=" * 70)
        print(f"  Control (全部模块): {results['control']['consciousness']:.4f}")
        print("-" * 70)
        print(f"  {'Module':<20} {'Consciousness':<15} {'Δ (drop)':<12} {'贡献%':<10}")
        print("-" * 70)

        sorted_ablations = sorted(
            results['ablations'].items(),
            key=lambda x: x[1]['delta'],
            reverse=True,
        )

        for name, data in sorted_ablations:
            delta_sign = '+' if data['delta'] < 0 else ' '
            print(f"  {name:<20} {data['consciousness']:<15.4f} "
                  f"{delta_sign}{-data['delta']:.4f}        "
                  f"{data['relative_contribution']*100:.1f}%")

        print("-" * 70)

        top_contributors = sorted_ablations[:3]
        if top_contributors:
            print(f"\n  意识水平最大贡献模块:")
            for i, (name, data) in enumerate(top_contributors):
                print(f"    {i+1}. {name}: "
                      f"移除后意识下降 {-data['delta']:.4f} "
                      f"({data['relative_contribution']*100:.1f}%)")

        print("=" * 70)

    def to_dataframe(self, results):
        """
        转换为 pandas DataFrame (如果可用)。
        """
        try:
            import pandas as pd
            rows = []
            for name, data in results['ablations'].items():
                rows.append({
                    'module': name,
                    'consciousness': data['consciousness'],
                    'delta': data['delta'],
                    'relative_contribution': data['relative_contribution'],
                })
            control_row = {
                'module': 'control',
                'consciousness': results['control']['consciousness'],
                'delta': 0.0,
                'relative_contribution': 0.0,
            }
            rows.append(control_row)
            return pd.DataFrame(rows).set_index('module')
        except ImportError:
            return None


class VirtualAblationStudy(AblationStudy):
    """
    虚拟摘除实验 — 不依赖 C++ CorticalBrain 的纯 Python 模拟，
    用于验证摘除框架本身的逻辑正确性。

    模拟一个简化的意识系统，其中各模块对意识水平有明确的已知贡献。
    """

    def __init__(self):
        super().__init__()
        self._active_modules = set()

    def _make_mock_brain(self):
        class MockBrain:
            def __init__(self, active_modules):
                self.active = active_modules
                self.step_count = 0

            def step(self, reward=0.0):
                self.step_count += 1

            def read_consciousness(self):
                class MockCS:
                    phi = 0.0
                    global_ignition = 0.0
                    self_prediction_error = 1.0
                cs = MockCS()
                if 'iit' in self.active:
                    cs.phi = 0.6
                if 'gwt' in self.active:
                    cs.global_ignition = 0.5
                if 'predictive' in self.active:
                    cs.self_prediction_error = 0.3
                return cs

            def get_first_person_salience(self):
                return 0.4 if 'qualia' in self.active else 0.0

            def get_perceptual_vividness(self):
                return 0.3 if 'qualia' in self.active else 0.0

        return MockBrain(self._active_modules)

    def run_mock_ablation(self):
        """
        运行模拟摘除实验，验证框架逻辑。
        返回每个模块移除后的意识水平。
        """
        all_modules = {'iit', 'gwt', 'predictive', 'qualia'}

        self._active_modules = all_modules.copy()
        self.brain_factory = self._make_mock_brain
        self.ablations = {}
        for mod in all_modules:
            remaining = all_modules - {mod}
            def make_disabler(keep=remaining):
                def disable(_brain=None):
                    class MockBrain:
                        def __init__(self):
                            self.active = keep
                            self.step_count = 0
                        def step(self, r=0.0):
                            self.step_count += 1
                        def read_consciousness(self):
                            class MockCS:
                                phi = 0.0
                                global_ignition = 0.0
                                self_prediction_error = 1.0
                            cs = MockCS()
                            if 'iit' in keep:
                                cs.phi = 0.6
                            if 'gwt' in keep:
                                cs.global_ignition = 0.5
                            if 'predictive' in keep:
                                cs.self_prediction_error = 0.3
                            return cs
                        def get_first_person_salience(self):
                            return 0.4 if 'qualia' in keep else 0.0
                        def get_perceptual_vividness(self):
                            return 0.3 if 'qualia' in keep else 0.0
                    return MockBrain()
                return disable
            self.ablations[mod] = make_disabler()

        return self.run_all(n_steps=10)


import pytest


class TestAblationStudyFramework:
    """验证摘除实验框架的逻辑正确性。"""

    def test_virtual_ablation_runs(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        assert 'control' in results
        assert 'ablations' in results
        assert len(results['ablations']) == 4

    def test_ablation_reduces_consciousness(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        control = results['control']['consciousness']
        for name, data in results['ablations'].items():
            assert data['consciousness'] < control, \
                f"摘除 {name} 应降低意识水平, 但 {data['consciousness']} >= {control}"

    def test_iit_ablation_is_largest_contributor(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        sorted_by_contribution = sorted(
            results['ablations'].items(),
            key=lambda x: x[1]['relative_contribution'],
            reverse=True,
        )
        top = sorted_by_contribution[0][0]
        assert top == 'iit', \
            f"IIT (权重0.35) 应贡献最大, 但 {top} 排第一"

    def test_ablation_delta_sum_positive(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        total_delta = sum(
            abs(data['delta']) for data in results['ablations'].values()
        )
        assert total_delta > 0, "摘除至少一个模块应产生可测量的 delta"

    def test_print_report_does_not_crash(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        study.print_report(results)

    def test_to_dataframe(self):
        study = VirtualAblationStudy()
        results = study.run_mock_ablation()
        df = study.to_dataframe(results)
        if df is not None:
            assert len(df) == 5
            assert 'control' in df.index

    def test_register_brain_factory_required(self):
        study = AblationStudy()
        with pytest.raises(ValueError):
            study.run_all()