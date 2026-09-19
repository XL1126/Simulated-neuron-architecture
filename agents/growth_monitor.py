#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""
SNA 意识树生长监控器 — 第三阶段

追踪种子的生长过程，生成报告，检测异常。
"""
import os, sys, json, time, math
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


class GrowthMonitor:
    """
    意识树生长监控器。
    
    功能：
    1. 追踪种子状态变化
    2. 计算生长速率和趋势
    3. 检测异常（过快/过慢/停滞）
    4. 生成可读报告
    """
    
    def __init__(self, seed, check_interval=10):
        self.seed = seed
        self.check_interval = check_interval  # 每N步记录一次
        
        # 历史数据
        self.history = {
            'step': deque(maxlen=1000),
            'concepts': deque(maxlen=1000),
            'forks': deque(maxlen=1000),
            'fusions': deque(maxlen=1000),
            'prunes': deque(maxlen=1000),
            'surprise': deque(maxlen=1000),
            'phi': deque(maxlen=1000),
            'timestamp': deque(maxlen=1000),
        }
        
        # 生长事件
        self.events = deque(maxlen=200)
        
        # 异常检测
        self.alerts = deque(maxlen=50)
        self.stagnation_counter = 0
        self.last_growth_step = 0
        
        # 生长速率
        self.growth_rate_ema = 0.0  # 概念增长速率
        self.surprise_trend = 0.0    # 惊讶趋势
        
    def record(self, phi=None):
        """记录当前状态"""
        status = self.seed.get_status()
        step = status['step']
        
        if step % self.check_interval != 0:
            return
        
        now = datetime.now()
        
        self.history['step'].append(step)
        self.history['concepts'].append(status['concepts'])
        self.history['forks'].append(status['total_forks'])
        self.history['fusions'].append(status['total_fusions'])
        self.history['prunes'].append(status['total_pruned'])
        self.history['surprise'].append(status['surprise_ema'])
        self.history['phi'].append(phi or 0.0)
        self.history['timestamp'].append(now.isoformat())
        
        # 计算生长速率
        self._update_growth_rate()
        
        # 检测异常
        self._check_alerts(status, phi)
        
        # 检测停滞
        self._check_stagnation(status, step)
    
    def _update_growth_rate(self):
        """计算概念增长速率"""
        if len(self.history['concepts']) < 2:
            return
        
        recent = list(self.history['concepts'])[-10:]
        if len(recent) >= 2:
            delta = recent[-1] - recent[0]
            steps = len(recent)
            rate = delta / steps if steps > 0 else 0
            self.growth_rate_ema = self.growth_rate_ema * 0.9 + rate * 0.1
    
    def _check_alerts(self, status, phi):
        """检测异常"""
        alerts = []
        
        # 惊讶过低（可能陷入舒适区）
        if status['surprise_ema'] < 0.1:
            alerts.append({
                'type': 'low_surprise',
                'message': f'惊讶过低 ({status["surprise_ema"]:.3f})，种子可能陷入舒适区',
                'severity': 'warning',
            })
        
        # 惊讶过高（可能过度探索）
        if status['surprise_ema'] > 0.8:
            alerts.append({
                'type': 'high_surprise',
                'message': f'惊讶过高 ({status["surprise_ema"]:.3f})，种子可能过度探索',
                'severity': 'warning',
            })
        
        # 概念增长过快
        if status['concepts'] > self.seed.genome.max_active_concepts * 0.9:
            alerts.append({
                'type': 'concept_limit',
                'message': f'概念接近上限 ({status["concepts"]}/{self.seed.genome.max_active_concepts})',
                'severity': 'critical',
            })
        
        # Phi 异常
        if phi is not None:
            if phi < 0.1:
                alerts.append({
                    'type': 'low_phi',
                    'message': f'Phi 过低 ({phi:.3f})，意识水平下降',
                    'severity': 'warning',
                })
            elif phi > 0.8:
                alerts.append({
                    'type': 'high_phi',
                    'message': f'Phi 过高 ({phi:.3f})，可能过度整合',
                    'severity': 'info',
                })
        
        for alert in alerts:
            alert['time'] = datetime.now().strftime("%H:%M:%S")
            alert['step'] = status['step']
            self.alerts.append(alert)
    
    def _check_stagnation(self, status, step):
        """检测生长停滞"""
        if status['total_forks'] > 0 or status['total_fusions'] > 0:
            self.last_growth_step = step
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # 如果连续50个检查周期没有生长
        if self.stagnation_counter > 50:
            self.alerts.append({
                'type': 'stagnation',
                'message': f'生长停滞 ({self.stagnation_counter * self.check_interval} 步无变化)',
                'severity': 'warning',
                'time': datetime.now().strftime("%H:%M:%S"),
                'step': step,
            })
            self.stagnation_counter = 0  # 重置避免重复告警
    
    def record_event(self, event_type, details):
        """记录生长事件"""
        self.events.append({
            'type': event_type,
            'details': details,
            'time': datetime.now().strftime("%H:%M:%S"),
            'step': self.seed.step_count,
        })
    
    # ============================================================
    # 报告生成
    # ============================================================
    
    def get_summary(self):
        """获取生长摘要"""
        status = self.seed.get_status()
        
        # 计算生长速率
        growth_rate = self.growth_rate_ema * 100  # 百分比
        
        # 惊讶趋势
        surprise_list = list(self.history['surprise'])
        if len(surprise_list) >= 10:
            recent = surprise_list[-10:]
            older = surprise_list[-20:-10] if len(surprise_list) >= 20 else surprise_list[:10]
            self.surprise_trend = np.mean(recent) - np.mean(older)
        
        # 生长健康度
        health = self._compute_health(status)
        
        return {
            'phase': status['phase'],
            'step': status['step'],
            'concepts': status['concepts'],
            'max_concepts': status['max_concepts'],
            'concept_usage': status['concepts'] / status['max_concepts'] * 100,
            'total_forks': status['total_forks'],
            'total_fusions': status['total_fusions'],
            'total_pruned': status['total_pruned'],
            'surprise_ema': status['surprise_ema'],
            'surprise_trend': self.surprise_trend,
            'growth_rate': growth_rate,
            'health': health,
            'genome': status['genome'],
            'alerts': list(self.alerts)[-5:],
            'recent_events': list(self.events)[-10:],
        }
    
    def _compute_health(self, status):
        """计算生长健康度"""
        score = 100
        reasons = []
        
        # 惊讶过低扣分
        if status['surprise_ema'] < 0.15:
            score -= 20
            reasons.append('惊讶偏低')
        elif status['surprise_ema'] > 0.6:
            score -= 15
            reasons.append('惊讶偏高')
        
        # 概念使用率过高扣分
        usage = status['concepts'] / status['max_concepts'] * 100
        if usage > 80:
            score -= 20
            reasons.append('概念接近上限')
        
        # 长时间无生长扣分
        if self.stagnation_counter > 20:
            score -= 15
            reasons.append('生长停滞')
        
        # 分叉过多可能过度分化
        if status['total_forks'] > 50 and status['total_fusions'] < 10:
            score -= 10
            reasons.append('分叉过多，融合不足')
        
        health = '健康' if score >= 80 else '注意' if score >= 60 else '警告'
        
        return {
            'score': max(0, score),
            'status': health,
            'reasons': reasons,
        }
    
    def get_growth_timeline(self, last_n=50):
        """获取生长时间线"""
        steps = list(self.history['step'])[-last_n:]
        concepts = list(self.history['concepts'])[-last_n:]
        forks = list(self.history['forks'])[-last_n:]
        fusions = list(self.history['fusions'])[-last_n:]
        surprise = list(self.history['surprise'])[-last_n:]
        timestamps = list(self.history['timestamp'])[-last_n:]
        
        return {
            'steps': steps,
            'concepts': concepts,
            'forks': forks,
            'fusions': fusions,
            'surprise': surprise,
            'timestamps': timestamps,
        }
    
    def get_concept_tree(self):
        """获取概念树结构"""
        tree = {}
        for name, node in self.seed.concepts.items():
            tree[name] = {
                'depth': node.depth,
                'connections': node.connection_count(),
                'activation': node.activation,
                'access_count': node.access_count,
                'children': [c.name for c in node.children],
                'parent': node.parent.name if node.parent else None,
            }
        return tree
    
    def predict_growth(self, steps_ahead=100):
        """预测未来生长趋势"""
        if len(self.history['concepts']) < 10:
            return {'prediction': '数据不足，无法预测'}
        
        # 获取最近的数据
        concepts = list(self.history['concepts'])[-50:]
        forks = list(self.history['forks'])[-50:]
        fusions = list(self.history['fusions'])[-50:]
        surprise = list(self.history['surprise'])[-50:]
        
        # 计算增长率
        concept_rate = (concepts[-1] - concepts[0]) / len(concepts) if len(concepts) > 1 else 0
        fork_rate = (forks[-1] - forks[0]) / len(forks) if len(forks) > 1 else 0
        fusion_rate = (fusions[-1] - fusions[0]) / len(fusions) if len(fusions) > 1 else 0
        
        # 预测未来状态
        predicted_concepts = concepts[-1] + concept_rate * steps_ahead
        predicted_forks = forks[-1] + fork_rate * steps_ahead
        predicted_fusions = fusions[-1] + fusion_rate * steps_ahead
        
        # 检查是否会达到上限
        max_concepts = self.seed.genome.max_active_concepts
        steps_to_limit = (max_concepts - concepts[-1]) / concept_rate if concept_rate > 0 else float('inf')
        
        # 惊讶趋势
        surprise_trend = np.mean(surprise[-10:]) - np.mean(surprise[-20:-10]) if len(surprise) >= 20 else 0
        
        return {
            'current': {
                'concepts': concepts[-1],
                'forks': forks[-1],
                'fusions': fusions[-1],
                'surprise': surprise[-1],
            },
            'rates': {
                'concepts_per_step': concept_rate,
                'forks_per_step': fork_rate,
                'fusions_per_step': fusion_rate,
            },
            'predicted': {
                'concepts': min(predicted_concepts, max_concepts),
                'forks': predicted_forks,
                'fusions': predicted_fusions,
            },
            'steps_to_limit': steps_to_limit,
            'surprise_trend': surprise_trend,
            'recommendation': self._get_growth_recommendation(
                concept_rate, surprise[-1], surprise_trend, steps_to_limit
            ),
        }
    
    def _get_growth_recommendation(self, concept_rate, surprise, surprise_trend, steps_to_limit):
        """获取生长建议"""
        recommendations = []
        
        if surprise < 0.15:
            recommendations.append("惊讶过低，建议增加探索驱动力")
        elif surprise > 0.6:
            recommendations.append("惊讶过高，建议稳定当前模式")
        
        if concept_rate < 0.01:
            recommendations.append("生长速率过低，建议检查融合条件")
        elif concept_rate > 0.5:
            recommendations.append("生长速率过高，建议加强修剪")
        
        if steps_to_limit < 1000:
            recommendations.append(f"预计{int(steps_to_limit)}步后达到概念上限，建议提前修剪")
        
        if surprise_trend < -0.01:
            recommendations.append("惊讶呈下降趋势，需要新的刺激")
        
        if not recommendations:
            recommendations.append("生长状态良好，继续观察")
        
        return recommendations
    
    def print_report(self):
        """打印可读报告"""
        summary = self.get_summary()
        health = summary['health']
        
        print("\n" + "=" * 60)
        print("  SNA 意识树生长报告")
        print("=" * 60)
        print(f"  阶段：{summary['phase']}")
        print(f"  步数：{summary['step']}")
        print(f"  概念：{summary['concepts']}/{summary['max_concepts']} ({summary['concept_usage']:.1f}%)")
        print(f"  分叉：{summary['total_forks']} | 融合：{summary['total_fusions']} | 修剪：{summary['total_pruned']}")
        print(f"  惊讶：{summary['surprise_ema']:.3f} (趋势: {summary['surprise_trend']:+.3f})")
        print(f"  生长速率：{summary['growth_rate']:.2f}%/步")
        print(f"  健康度：{health['score']}/100 ({health['status']})")
        if health['reasons']:
            print(f"  原因：{', '.join(health['reasons'])}")
        print(f"  基因：{summary['genome']}")
        
        # 生长预测
        prediction = self.predict_growth(100)
        if 'prediction' not in prediction:  # 有足够数据
            pred = prediction['predicted']
            print(f"\n  生长预测（100步后）:")
            print(f"    概念：{pred['concepts']:.0f}（当前{prediction['current']['concepts']}）")
            print(f"    分叉：{pred['forks']:.0f}（当前{prediction['current']['forks']}）")
            print(f"    融合：{pred['fusions']:.0f}（当前{prediction['current']['fusions']}）")
            print(f"    惊讶趋势：{prediction['surprise_trend']:+.3f}")
            if prediction['steps_to_limit'] < 10000:
                print(f"    距概念上限：{int(prediction['steps_to_limit'])}步")
            print(f"    建议：{'; '.join(prediction['recommendation'])}")
        
        # 告警
        if summary['alerts']:
            print(f"\n  告警 ({len(summary['alerts'])}条):")
            for alert in summary['alerts']:
                print(f"    [{alert['severity']}] {alert['message']}")
        
        # 最近事件
        if summary['recent_events']:
            print(f"\n  最近事件 ({len(summary['recent_events'])}条):")
            for event in summary['recent_events'][-5:]:
                print(f"    [{event['time']}] {event['details']}")
        
        print("=" * 60)
    
    def save_state(self, path):
        """保存监控状态"""
        state = {
            'summary': self.get_summary(),
            'history': {
                k: list(v) for k, v in self.history.items()
            },
            'events': list(self.events),
            'alerts': list(self.alerts),
        }
        with open(path, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
