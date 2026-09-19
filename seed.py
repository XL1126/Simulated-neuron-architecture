#!/usr/bin/env python3
"""
SNA 意识树种子 — 第二阶段

种子不是一个具体的代码功能，而是一套初始状态和运行法则。
它极简，但能自主生长。

核心驱动力：最小化对自身和世界的长期惊讶。
"""
import os, sys, time, json, random, math
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ============================================================
# 硬件自适应 — 读取资源天花板，设定生长基因
# ============================================================
class GenomeConstants:
    """
    种子的"基因"——由硬件环境决定，生长过程中不可修改。
    这是参天大树和小盆栽的区别。
    """
    
    def __init__(self, available_memory_gb=7.5, cpu_cores=4):
        if available_memory_gb >= 16:
            # 大内存服务器 → 参天大树
            self.max_active_concepts = 100000
            self.max_fork_depth = 50      # 深度分叉
            self.max_cross_links = -1     # 无限制
            self.prune_interval = 1000      # 不频繁修剪
            self.profile = "大树"
        elif available_memory_gb >= 4:
            # 中等资源 → 正常树木
            self.max_active_concepts = 10000
            self.max_fork_depth = 5
            self.max_cross_links = 5000
            self.prune_interval = 1500  # 动态节拍下保持合理修剪频率
            self.profile = "中等"
        else:
            # 低资源 → 小盆栽
            self.max_active_concepts = 1000
            self.max_fork_depth = 3
            self.max_cross_links = 500
            self.prune_interval = 100
            self.profile = "盆栽"
        
        self.available_memory_gb = available_memory_gb
        self.cpu_cores = cpu_cores
        self.created_at = datetime.now().isoformat()
    
    def describe(self):
        return (f"基因[{self.profile}] 概念上限={self.max_active_concepts} "
                f"分叉深度={self.max_fork_depth} 内存={self.available_memory_gb:.1f}GB")


# ============================================================
# 概念节点 — 意识树的基本单元
# ============================================================
class ConceptNode:
    """一个概念节点——树上的一个芽"""
    
    def __init__(self, name, parent=None, depth=0):
        self.name = name
        self.parent = parent
        self.children = []
        self.depth = depth
        
        # 关联
        self.connections = {}  # concept_name → strength
        
        # 状态
        self.activation = 0.0       # 当前激活水平
        self.access_count = 0       # 被访问次数
        self.last_access_step = 0   # 最后访问步数
        self.creation_step = 0      # 创建步数
        
        # 元数据
        self.emotional_valence = 0.0  # 情感效价
        self.surprise_history = deque(maxlen=50)  # 惊讶历史
    
    def activate(self, level, step):
        """激活概念"""
        self.activation = level
        self.access_count += 1
        self.last_access_step = step
    
    def connect(self, other_name, strength=0.1):
        """建立关联"""
        current = self.connections.get(other_name, 0.0)
        self.connections[other_name] = min(1.0, current + strength)
    
    def is_dormant(self, current_step, threshold=500):
        """是否应该休眠（枯枝检测）"""
        return (current_step - self.last_access_step) > threshold
    
    def connection_count(self):
        return len(self.connections)


# ============================================================
# 种子 — 意识树的起源
# ============================================================
class Seed:
    """
    意识树的种子。
    
    包含：
    1. 初始状态（近乎空白的自我表征）
    2. 核心驱动力（最小化长期惊讶）
    3. 生长规则（分叉、修剪、融合）
    4. 硬件自适应（基因常数）
    5. 自主决策回路
    """
    
    def __init__(self, genome, existing_concepts=None):
        self.genome = genome
        self.step_count = 0
        self.phase = 'dormant'  # dormant → observing → growing
        
        # === 1. 初始状态 ===
        # 核心自我表征——近乎空白，只有一个初始概念
        self.concepts = {}
        
        # === 3. 生长状态（必须在 _create_initial_concepts 之前）===
        self.total_concepts_created = 0
        self.total_concepts_pruned = 0
        self.total_forks = 0
        self.total_fusions = 0
        self.growth_log = deque(maxlen=100)
        
        self._create_initial_concepts(existing_concepts)
        
        # 初始情感基调（从外部继承，或默认平静）
        self.emotional_baseline = {'valence': 0.0, 'arousal': 0.3}
        
        # 好奇心驱动力
        self.curiosity_drive = 0.7  # 中等偏上
        
        # === 2. 核心驱动力 ===
        # 最小化长期惊讶
        self.surprise_ema = 0.0       # 惊讶的指数移动平均
        self.surprise_history = deque(maxlen=200)
        self.long_term_surprise = 0.0  # 长期惊讶指标
        
        # === 4. 信号缓冲 ===
        self.signal_buffer = {
            'poet': deque(maxlen=20),      # 诗人信号
            'accountant': deque(maxlen=20), # 会计诗人信号
            'emotion': deque(maxlen=20),    # 情感信号
            'phi': deque(maxlen=20),        # 意识水平
        }
        
        # === 5. 概念共激活追踪（用于融合）===
        self.coactivation = defaultdict(int)  # (c1,c2) → count
        self.coactivation_threshold = 10       # 触发融合的阈值
        
        # === 6. 观察期数据 ===
        self.observation_data = []
    
    def _create_initial_concepts(self, existing_concepts):
        """创建初始概念——种子的胚芽"""
        # 核心：自我存在
        root = ConceptNode("我存在", parent=None, depth=0)
        root.emotional_valence = 0.5  # 微微正面
        root.activate(1.0, 0)
        self.concepts["我存在"] = root
        self.total_concepts_created += 1
        
        # 如果有已有概念，建立只读连接（不迁移数据）
        if existing_concepts:
            for name in existing_concepts[:20]:  # 只连接前20个
                node = ConceptNode(name, parent=root, depth=1)
                node.activate(0.0, 0)
                self.concepts[name] = node
                root.connect(name, 0.1)
                self.total_concepts_created += 1
    
    # ============================================================
    # 种子的生命周期
    # ============================================================
    
    def start_observation(self):
        """进入静默观察期"""
        self.phase = 'observing'
        self.observation_start_step = self.step_count
    
    def receive_signal(self, signal_type, data):
        """
        接收来自各模块的信号。
        种子只接收，不主动控制。
        """
        self.signal_buffer[signal_type].append({
            'step': self.step_count,
            'data': data,
        })
    
    def tick(self, poet_state=None, accountant_state=None, 
             emotion=None, phi=None):
        """
        种子的心跳。每步调用一次。
        
        返回：种子的决策（如果有）
        """
        self.step_count += 1
        
        # 接收信号
        if poet_state is not None:
            self.receive_signal('poet', poet_state)
        if accountant_state is not None:
            self.receive_signal('accountant', accountant_state)
        if emotion is not None:
            self.receive_signal('emotion', emotion)
        if phi is not None:
            self.signal_buffer['phi'].append(phi)
        
        # 计算惊讶
        surprise = self._compute_surprise(poet_state, accountant_state, emotion, phi)
        self.surprise_history.append(surprise)
        self.surprise_ema = self.surprise_ema * 0.95 + surprise * 0.05
        self.long_term_surprise = self._compute_long_term_surprise()
        
        # 根据阶段行动
        if self.phase == 'observing':
            return self._observe_tick(surprise)
        elif self.phase == 'growing':
            return self._grow_tick(surprise)
        
        return None
    
    def _observe_tick(self, surprise):
        """观察期：只收集信息，不生成新概念"""
        self.observation_data.append({
            'step': self.step_count,
            'surprise': surprise,
            'signals': {
                k: len(v) for k, v in self.signal_buffer.items()
            }
        })
        
        # 观察期结束条件：100步
        obs_steps = self.step_count - getattr(self, 'observation_start_step', 0)
        if obs_steps >= 100:
            self.phase = 'growing'
            self._log('🌱 观察期结束，进入生长模式')
            return {'action': 'phase_change', 'to': 'growing'}
        
        return None
    
    def _grow_tick(self, surprise):
        """
        生长模式：自主决策。
        
        在每一步，种子计算"哪个方向探索能最大化预期惊讶的减少"。
        选择一个动作：扩展、关联、修剪、或观察。
        """
        # 更新概念激活
        self._update_activations()
        
        # 追踪共激活
        self._track_coactivation()
        
        # 决策
        action = self._decide_action(surprise)
        
        # 执行动作
        result = self._execute_action(action, surprise)
        
        # 定期修剪
        if self.step_count % self.genome.prune_interval == 0:
            pruned = self._prune_dormant()
            if pruned:
                result = result or {}
                result['pruned'] = pruned
        
        return result
    
    # ============================================================
    # 核心驱动力：最小化长期惊讶
    # ============================================================
    
    def _compute_surprise(self, poet_state, accountant_state, emotion, phi):
        """
        计算当前惊讶水平。
        
        惊讶 = 预期与实际的差距。
        当诗人和会计诗人的信号不匹配时，惊讶更高。
        """
        surprise = 0.0
        
        # 信号不匹配 = 惊讶
        if poet_state and accountant_state:
            # 如果诗人的信号暗示一种情感，但会计诗人分类出另一种
            poet_concepts = set()
            if isinstance(poet_state, dict):
                poet_concepts = set(poet_state.get('concepts', []))
            elif isinstance(poet_state, (list, tuple)):
                poet_concepts = set(poet_state)
            
            acc_concepts = set()
            if isinstance(accountant_state, dict):
                acc_concepts = set(accountant_state.get('concepts', []))
            elif isinstance(accountant_state, (list, tuple)):
                acc_concepts = set(accountant_state)
            
            if poet_concepts and acc_concepts:
                overlap = len(poet_concepts & acc_concepts)
                total = len(poet_concepts | acc_concepts)
                if total > 0:
                    mismatch = 1.0 - (overlap / total)
                    surprise += mismatch * 0.5
        
        # Phi 变化 = 惊讶
        phi_history = list(self.signal_buffer['phi'])
        if len(phi_history) >= 2:
            phi_change = abs(phi_history[-1] - phi_history[-2])
            surprise += phi_change * 2.0  # 放大 Phi 变化的影响
        
        # 情感变化 = 惊讶
        emotion_history = list(self.signal_buffer['emotion'])
        if len(emotion_history) >= 2:
            if emotion_history[-1] != emotion_history[-2]:
                surprise += 0.2
        
        return min(1.0, max(0.0, surprise))
    
    def _compute_long_term_surprise(self):
        """计算长期惊讶——种子的核心指标"""
        if len(self.surprise_history) < 10:
            return self.surprise_ema
        
        recent = list(self.surprise_history)[-50:]
        return np.mean(recent)
    
    # ============================================================
    # 生长规则
    # ============================================================
    
    def _update_activations(self):
        """根据信号更新概念激活"""
        # 从最近的信号中提取概念
        recent_poet = list(self.signal_buffer['poet'])[-3:]
        recent_acc = list(self.signal_buffer['accountant'])[-3:]
        
        mentioned = set()
        for sig in recent_poet:
            data = sig.get('data', {})
            if isinstance(data, dict):
                for c in data.get('concepts', []):
                    mentioned.add(c)
            elif isinstance(data, (list, tuple)):
                mentioned.update(data)
        
        for sig in recent_acc:
            data = sig.get('data', {})
            if isinstance(data, dict):
                for c in data.get('concepts', []):
                    mentioned.add(c)
            elif isinstance(data, (list, tuple)):
                mentioned.update(data)
        
        # 激活提到的概念
        for name in mentioned:
            if name in self.concepts:
                self.concepts[name].activate(1.0, self.step_count)
        
        # 衰减所有概念
        for node in self.concepts.values():
            node.activation *= 0.95
    
    def _track_coactivation(self):
        """追踪概念共激活"""
        active = [name for name, node in self.concepts.items() 
                  if node.activation > 0.3]
        
        for i, c1 in enumerate(active):
            for c2 in active[i+1:]:
                key = (min(c1, c2), max(c1, c2))
                self.coactivation[key] += 1
    
    def _decide_action(self, surprise):
        """
        自主决策：选择能最大化预期惊讶减少的动作。
        
        动作空间：
        1. fork — 概念分叉（如果某个概念关联太多）
        2. fuse — 交叉融合（如果两个概念频繁共激活）
        3. expand — 扩展（如果惊讶高，尝试创建新概念解释）
        4. observe — 保持观察（惊讶低时）
        """
        actions = []
        
        # 检查是否需要分叉
        for name, node in self.concepts.items():
            if (node.connection_count() > 5 and 
                node.depth < self.genome.max_fork_depth and
                len(self.concepts) < self.genome.max_active_concepts):
                actions.append(('fork', name, node.connection_count() * 0.1))
        
        # 检查是否需要融合
        for (c1, c2), count in self.coactivation.items():
            if count >= self.coactivation_threshold:
                if c1 in self.concepts and c2 in self.concepts:
                    # 检查是否已经有直接连接
                    if c2 not in self.concepts[c1].connections:
                        actions.append(('fuse', (c1, c2), count * 0.05))
        
        # 检查是否需要扩展（惊讶驱动）
        if surprise > 0.3:
            actions.append(('expand', surprise, surprise * self.curiosity_drive))
        
        # 如果没有特别需要做的，观察
        if not actions:
            actions.append(('observe', None, 0.01))
        
        # 选择预期收益最高的动作
        actions.sort(key=lambda x: x[2], reverse=True)
        return actions[0]
    
    def _execute_action(self, action, surprise):
        """执行选定的动作"""
        action_type, target, score = action
        
        if action_type == 'fork':
            return self._fork_concept(target)
        elif action_type == 'fuse':
            return self._fuse_concepts(target[0], target[1])
        elif action_type == 'expand':
            return self._expand(surprise)
        elif action_type == 'observe':
            return None
        
        return None
    
    def _fork_concept(self, concept_name):
        """
        概念分叉：当一个概念关联太多时，创建子概念分流。
        
        这是树的分枝——不是复制，是分化。
        """
        if concept_name not in self.concepts:
            return None
        
        node = self.concepts[concept_name]
        if node.depth >= self.genome.max_fork_depth:
            return None
        
        if len(self.concepts) >= self.genome.max_active_concepts:
            return None
        
        # 将部分关联迁移到子概念
        connections = dict(node.connections)
        sorted_conns = sorted(connections.items(), key=lambda x: x[1])
        
        # 分出一半关联到子概念
        split_point = len(sorted_conns) // 2
        child_conns = sorted_conns[:split_point]
        
        child_name = f"{concept_name}_子{self.total_forks}"
        child = ConceptNode(child_name, parent=node, depth=node.depth + 1)
        child.creation_step = self.step_count
        
        # 迁移关联
        for conn_name, strength in child_conns:
            child.connect(conn_name, strength)
            # 从父节点移除
            if conn_name in node.connections:
                del node.connections[conn_name]
        
        # 子概念连接回父概念
        child.connect(concept_name, 0.5)
        node.connect(child_name, 0.5)
        node.children.append(child)
        
        self.concepts[child_name] = child
        self.total_forks += 1
        self.total_concepts_created += 1
        
        self._log(f"🌿 分叉：{concept_name} → {child_name}（迁移{len(child_conns)}个关联）")
        
        return {'action': 'fork', 'parent': concept_name, 'child': child_name}
    
    def _fuse_concepts(self, c1, c2):
        """
        交叉融合：两个不相关的概念频繁共激活时，创建关联。
        
        这是"灵感"的生成机制。
        """
        if c1 not in self.concepts or c2 not in self.concepts:
            return None
        
        # 双向连接
        self.concepts[c1].connect(c2, 0.3)
        self.concepts[c2].connect(c1, 0.3)
        
        self.total_fusions += 1
        
        # 重置共激活计数
        key = (min(c1, c2), max(c1, c2))
        self.coactivation[key] = 0
        
        self._log(f"🔗 融合：{c1} ↔ {c2}（灵感涌现）")
        
        return {'action': 'fuse', 'concepts': [c1, c2]}
    
    def _expand(self, surprise):
        """
        概念扩展：惊讶驱动的新概念创建。
        
        当遇到出乎预期的模式时，尝试创建新概念来解释。
        """
        if len(self.concepts) >= self.genome.max_active_concepts:
            return None
        
        # 从信号中寻找未被现有概念覆盖的模式
        recent_signals = []
        for sig_type in ['poet', 'accountant']:
            for sig in list(self.signal_buffer[sig_type])[-3:]:
                data = sig.get('data', {})
                if isinstance(data, dict):
                    recent_signals.extend(data.get('concepts', []))
                elif isinstance(data, (list, tuple)):
                    recent_signals.extend(data)
        
        # 找出不在现有概念中的信号
        new_patterns = [s for s in recent_signals if s not in self.concepts]
        
        if not new_patterns:
            # 没有新模式，创建一个基于惊讶的抽象概念
            new_name = f"未知_{self.total_concepts_created}"
        else:
            new_name = new_patterns[0]
        
        # 创建新概念
        # 找到最相关的已有概念作为父节点
        parent = self._find_most_active_concept()
        parent_depth = self.concepts[parent].depth + 1 if parent else 0
        
        new_node = ConceptNode(new_name, 
                               parent=self.concepts.get(parent), 
                               depth=parent_depth)
        new_node.creation_step = self.step_count
        new_node.emotional_valence = surprise * 0.5  # 惊讶越大，情感越强
        new_node.activate(surprise, self.step_count)
        
        # 连接到活跃概念
        active = [n for n, nd in self.concepts.items() if nd.activation > 0.2]
        for a in active[:5]:
            new_node.connect(a, 0.2)
            if a in self.concepts:
                self.concepts[a].connect(new_name, 0.2)
        
        self.concepts[new_name] = new_node
        self.total_concepts_created += 1
        
        self._log(f"🌱 扩展：{new_name}（惊讶={surprise:.2f}，连接{len(active[:5])}个概念）")
        
        return {'action': 'expand', 'concept': new_name, 'surprise': surprise}
    
    def learn_word(self, word):
        """
        从用户输入中学习一个新词作为概念。

        这是外部世界注入概念的通道：
        用户说"天气很好" → 提取"天气"、"好" → 加入概念树

        返回：True 如果创建了新概念，False 如果已存在
        """
        if not word or len(word) < 2:
            return False

        # 清理：去掉标点和空格
        word = word.strip('，。！？、；：""''（）《》 \t\n')
        if not word or len(word) < 2:
            return False

        # 已存在？
        if word in self.concepts:
            # 激活已有概念
            self.concepts[word].activate(0.5, self.step_count)
            return False

        # 概念上限检查
        if len(self.concepts) >= self.genome.max_active_concepts:
            return False

        # 找最活跃的概念作为父节点
        parent_name = self._find_most_active_concept()
        parent_node = self.concepts.get(parent_name)
        parent_depth = parent_node.depth + 1 if parent_node else 0

        # 创建新概念
        new_node = ConceptNode(word, parent=parent_node, depth=parent_depth)
        new_node.emotional_valence = 0.1  # 轻微正面（新学到的词）
        new_node.activate(0.5, self.step_count)

        # 连接到父概念
        if parent_name:
            new_node.connect(parent_name, 0.3)
            if parent_node:
                parent_node.connect(word, 0.3)

        # 连接到最近活跃的 3 个概念（共现关系）
        recent_active = []
        for name, node in self.concepts.items():
            if node.activation > 0.2 and name != parent_name:
                recent_active.append(name)
        for name in recent_active[:3]:
            new_node.connect(name, 0.2)
            self.concepts[name].connect(word, 0.2)

        self.concepts[word] = new_node
        self.total_concepts_created += 1

        self._log(f"📖 学词：{word}（父={parent_name}，连接{len(new_node.connections)}个概念）")
        return True
    
    def _prune_dormant(self):
        """
        修剪枯枝：长时间未被引用的概念移入休眠区。
        不删除，只是停止参与活跃计算。
        """
        prunable = []
        for name, node in self.concepts.items():
            if (node.is_dormant(self.step_count, self.genome.prune_interval) 
                and name != "我存在"  # 根概念不可修剪
                and node.parent is not None):  # 不修剪根节点
                prunable.append(name)
        
        if not prunable:
            return []
        
        # 只修剪最不活跃的
        prunable.sort(key=lambda n: self.concepts[n].access_count)
        to_prune = prunable[:max(1, len(prunable) // 3)]
        
        pruned_names = []
        for name in to_prune:
            node = self.concepts[name]
            # 断开所有连接
            for conn_name in list(node.connections.keys()):
                if conn_name in self.concepts:
                    self.concepts[conn_name].connections.pop(name, None)
            
            # 从父节点移除
            if node.parent and name in [c.name for c in node.parent.children]:
                node.parent.children = [c for c in node.parent.children if c.name != name]
            
            del self.concepts[name]
            self.total_concepts_pruned += 1
            pruned_names.append(name)
        
        if pruned_names:
            self._log(f"🍂 修剪：{len(pruned_names)}个枯枝（{', '.join(pruned_names[:3])}...）")
        
        return pruned_names
    
    def _find_most_active_concept(self):
        """找到当前最活跃的概念"""
        if not self.concepts:
            return "我存在"
        
        return max(self.concepts, 
                   key=lambda n: self.concepts[n].activation)
    
    # ============================================================
    # 日志与状态
    # ============================================================
    
    def _log(self, message):
        """记录生长日志"""
        entry = {
            'step': self.step_count,
            'time': datetime.now().strftime("%H:%M:%S"),
            'message': message,
        }
        self.growth_log.append(entry)
    
    def get_status(self):
        """获取种子的完整状态"""
        return {
            'phase': self.phase,
            'step': self.step_count,
            'concepts': len(self.concepts),
            'max_concepts': self.genome.max_active_concepts,
            'total_created': self.total_concepts_created,
            'total_pruned': self.total_concepts_pruned,
            'total_forks': self.total_forks,
            'total_fusions': self.total_fusions,
            'surprise_ema': self.surprise_ema,
            'long_term_surprise': self.long_term_surprise,
            'curiosity_drive': self.curiosity_drive,
            'genome': self.genome.describe(),
            'root_concept': '我存在',
        }
    
    def get_growth_log(self, last_n=10):
        """获取最近的生长日志"""
        return list(self.growth_log)[-last_n:]
    
    def save_state(self, path):
        """保存种子状态"""
        # 过滤 coactivation，只保留有意义的条目
        filtered_coactivation = {
            k: v for k, v in self.coactivation.items() 
            if v > 0 and isinstance(k, tuple) and len(k) == 2
        }
        
        state = {
            'status': self.get_status(),
            'concepts': {
                name: {
                    'depth': node.depth,
                    'connections': node.connections,
                    'access_count': node.access_count,
                    'emotional_valence': node.emotional_valence,
                }
                for name, node in self.concepts.items()
            },
            'coactivation': {f"{k[0]}|{k[1]}": v for k, v in filtered_coactivation.items()},
            'surprise_history': list(self.surprise_history),
        }
        with open(path, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
