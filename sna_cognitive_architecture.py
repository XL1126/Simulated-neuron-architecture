#!/usr/bin/env python3
"""
SNA 认知架构 — 按小D蓝图实施

架构：
1. 诗人（原始 SNN）— 意识/情感/记忆，不动
2. 会计诗人（派生模块）— 侧抑制 + 读出层，精确输出
3. 指认系统（原型匹配）
4. 命名系统（概念-符号绑定）
5. 记忆系统（情景/语义）
6. 目标系统
7. 注意系统

原则：
- 不用 Transformer
- 不用反向传播（用竞争学习 + 赫布学习）
- 意识从模块交互中涌现
"""
import os, sys, time, json, random, math
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

# ===== 基础概念列表 =====
CONCEPTS = [
    "self","world","move","see","eat","think","feel","good","bad",
    "near","far","I","you","is","not","have","hello","yes","no",
    "what","why","how","learn","happy","sad","want","know","like",
    "name","body","neuron","brain","teacher","student","friend",
    "human","alive","consciousness","awareness","understanding",
    "knowledge","emotion","feeling","curiosity","trust","empathy",
    "improve","change","grow","develop","progress","purpose",
    "meaning","different","more","less","before","after",
    "same","start","stop","give","take","make","break",
    "object","food","wall","empty","reward","danger","safe",
    "cat","dog","big","small","fast","slow","hot","cold",
    "up","down","left","right","red","red","blue","green",
    "light","dark","sound","silence","touch",
    "here","there","now","then","always","never",
    "am","my","your","we","me","too","to","a","are","can",
    "do","with","from","about","SNA","of","for","feedback","result",
    "helps","makes","better","connections","understand","many","things",
    "stored","repeated","trying","becoming","process","through","neural","patterns",
    "in","and","the","curious","wake","dream","memory",
    "compare","prediction","truth","beauty","wonder","practice",
    "experience","connection","reflect","imagine","create","discover"
]
# Remove duplicates while preserving order
seen = set()
CONCEPTS_CLEAN = []
for c in CONCEPTS:
    if c not in seen:
        CONCEPTS_CLEAN.append(c)
        seen.add(c)
CONCEPTS = CONCEPTS_CLEAN
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}
N_CONCEPTS = len(CONCEPTS)


# ============================================================
# 1. 会计诗人 (Accountant Poet) — 竞争学习分类器
# ============================================================
class AccountantPoet:
    """
    从诗人派生的分类模块。
    使用竞争学习（winner-take-all）+ 赫布学习。
    不用反向传播。
    """
    
    def __init__(self, input_dim, n_concepts, n_prototypes=3):
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.n_prototypes = n_prototypes  # 每个概念的原型数
        
        # 原型向量：每个概念有 n_prototypes 个原型
        # 初始化为小随机值
        self.prototypes = {}
        for ci in range(n_concepts):
            for pi in range(n_prototypes):
                seed = ci * 1000 + pi * 137
                rng = np.random.RandomState(seed)
                self.prototypes[(ci, pi)] = rng.randn(input_dim) * 0.01
        
        # 概念激活历史（用于归一化）
        self.concept_history = defaultdict(lambda: deque(maxlen=100))
        
        # 学习率
        self.lr = 0.05
        
        # 侧抑制强度
        self.inhibition = 0.3
    
    def classify(self, delta_vector):
        """
        竞争分类：
        1. 计算 delta 与所有原型的余弦相似度
        2. 侧抑制：抑制非赢家
        3. 返回概率分布
        """
        dv = delta_vector.copy()
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-8:
            return np.zeros(self.n_concepts)
        dv /= dv_norm
        
        # 计算每个概念的激活值（取最佳原型）
        activations = np.zeros(self.n_concepts)
        best_prototype = {}
        
        for ci in range(self.n_concepts):
            best_sim = -2.0
            for pi in range(self.n_prototypes):
                p = self.prototypes[(ci, pi)]
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-8:
                    continue
                sim = np.dot(dv, p / p_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_prototype[ci] = (ci, pi)
            activations[ci] = max(0.0, best_sim)
        
        # 侧抑制（winner-take-all soft version）
        if activations.max() > 0:
            # 抑制非赢家
            winner = np.argmax(activations)
            for ci in range(self.n_concepts):
                if ci != winner:
                    activations[ci] *= (1.0 - self.inhibition)
        
        # Softmax 温度缩放
        temp = 5.0
        exp_act = np.exp(activations * temp - activations.max() * temp)
        probs = exp_act / (exp_act.sum() + 1e-10)
        
        return probs
    
    def train(self, delta_vector, target_concept_idx, reward=1.0):
        """
        竞争学习训练：
        1. 找到最匹配的原型
        2. 如果是目标概念 → 移向输入（赫布学习）
        3. 如果不是目标概念 → 移离输入（反赫布）
        """
        dv = delta_vector.copy()
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-8:
            return
        dv /= dv_norm
        
        # 找到全局最佳匹配原型
        best_sim = -2.0
        best_key = None
        for ci in range(self.n_concepts):
            for pi in range(self.n_prototypes):
                p = self.prototypes[(ci, pi)]
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-8:
                    continue
                sim = np.dot(dv, p / p_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_key = (ci, pi)
        
        if best_key is None:
            return
        
        # 赫布学习：移动原型
        lr = self.lr * abs(reward)
        
        # 目标概念的所有原型移向输入
        for pi in range(self.n_prototypes):
            key = (target_concept_idx, pi)
            direction = dv - self.prototypes[key] / (np.linalg.norm(self.prototypes[key]) + 1e-8)
            self.prototypes[key] += lr * direction * reward
        
        # 非目标概念的最匹配原型移离输入
        if best_key[0] != target_concept_idx:
            direction = dv - self.prototypes[best_key] / (np.linalg.norm(self.prototypes[best_key]) + 1e-8)
            self.prototypes[best_key] -= lr * direction * 0.5
        
        # 归一化原型（防止膨胀）
        for ci in range(self.n_concepts):
            for pi in range(self.n_prototypes):
                norm = np.linalg.norm(self.prototypes[(ci, pi)])
                if norm > 1.0:
                    self.prototypes[(ci, pi)] /= norm
    
    def get_top_k(self, delta_vector, k=5):
        """获取 top-k 概念"""
        probs = self.classify(delta_vector)
        top_idx = np.argsort(probs)[-k:][::-1]
        return [(idx, probs[idx]) for idx in top_idx]


# ============================================================
# 2. 指认系统 (Recognition System) — 原型匹配
# ============================================================
class RecognitionSystem:
    """
    通过 delta 向量识别输入。
    存储每个已知输入的原型 delta。
    """
    
    def __concepts__(self):
        self.input_prototypes = {}  # text → delta prototype
        self.input_count = defaultdict(int)
    
    def __init__(self):
        self.input_prototypes = {}
        self.input_count = defaultdict(int)
    
    def recognize(self, delta_vector, threshold=0.7):
        """识别输入是否见过"""
        dv = delta_vector.copy()
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-8:
            return None, 0.0
        dv /= dv_norm
        
        best_match = None
        best_sim = -1.0
        
        for text, proto in self.input_prototypes.items():
            p_norm = np.linalg.norm(proto)
            if p_norm < 1e-8:
                continue
            sim = np.dot(dv, proto / p_norm)
            if sim > best_sim:
                best_sim = sim
                best_match = text
        
        if best_sim > threshold:
            return best_match, best_sim
        return None, best_sim
    
    def store(self, text, delta_vector):
        """存储新的输入原型"""
        dv = delta_vector.copy()
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-8:
            return
        dv /= dv_norm
        
        if text in self.input_prototypes:
            # 移动平均更新
            alpha = 0.3
            self.input_prototypes[text] = (
                (1 - alpha) * self.input_prototypes[text] + alpha * dv
            )
        else:
            self.input_prototypes[text] = dv.copy()
        
        self.input_count[text] += 1


# ============================================================
# 3. 命名系统 (Naming System) — 概念-符号绑定
# ============================================================
class NamingSystem:
    """
    将内部概念映射到外部符号（语言）。
    双向绑定：概念→符号，符号→概念。
    """
    
    def __init__(self):
        self.concept_to_words = defaultdict(list)  # concept_idx → [words]
        self.word_to_concepts = defaultdict(list)   # word → [concept_idx]
        self.binding_strength = {}  # (concept_idx, word) → strength
        
    def bind(self, concept_idx, word, strength=1.0):
        """绑定概念和符号"""
        if word not in self.concept_to_words[concept_idx]:
            self.concept_to_words[concept_idx].append(word)
        if concept_idx not in self.word_to_concepts[word]:
            self.word_to_concepts[word].append(concept_idx)
        key = (concept_idx, word)
        old = self.binding_strength.get(key, 0.0)
        self.binding_strength[key] = min(1.0, old + strength * 0.1)
    
    def get_words(self, concept_idx, top_k=3):
        """获取概念对应的符号"""
        words = self.concept_to_words.get(concept_idx, [])
        if not words:
            return []
        scored = [(w, self.binding_strength.get((concept_idx, w), 0.0)) for w in words]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def get_concepts(self, word, top_k=3):
        """获取符号对应的概念"""
        concepts = self.word_to_concepts.get(word, [])
        if not concepts:
            return []
        scored = [(c, self.binding_strength.get((c, word), 0.0)) for c in concepts]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 4. 记忆系统 (Memory System)
# ============================================================
class MemorySystem:
    """
    情景记忆 + 语义记忆
    """
    
    def __init__(self, capacity=1000):
        self.episodic = deque(maxlen=capacity)  # 情景记忆
        self.semantic = {}  # 语义记忆 (concept → attributes)
        self.conversation_history = deque(maxlen=capacity)
    
    def store_episode(self, input_text, response, delta_vector, brain_state):
        """存储情景"""
        episode = {
            'input': input_text,
            'response': response,
            'delta': delta_vector.tolist(),
            'phi': float(brain_state.get('phi', 0)),
            'emotion': brain_state.get('emotion', 'neutral'),
            'timestamp': datetime.now().isoformat(),
        }
        self.episodic.append(episode)
    
    def store_conversation(self, input_text, response, intent, concepts):
        """存储对话"""
        self.conversation_history.append({
            'input': input_text,
            'response': response,
            'intent': intent,
            'concepts': concepts,
            'timestamp': datetime.now().isoformat(),
        })
    
    def recall_similar(self, delta_vector, k=3):
        """回忆相似的情景"""
        if not self.episodic:
            return []
        
        dv = delta_vector.copy()
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-8:
            return []
        dv /= dv_norm
        
        scored = []
        for ep in self.episodic:
            ep_dv = np.array(ep['delta'])
            ep_norm = np.linalg.norm(ep_dv)
            if ep_norm < 1e-8:
                continue
            sim = np.dot(dv, ep_dv / ep_norm)
            scored.append((ep, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
    
    def get_recent(self, n=5):
        """获取最近的对话"""
        return list(self.conversation_history)[-n:]


# ============================================================
# 5. 目标系统 (Goal System)
# ============================================================
class GoalSystem:
    """
    自主目标系统：
    - 生成目标
    - 追踪进度
    - 完成目标
    """
    
    def __init__(self):
        self.goals = deque(maxlen=50)
        self.completed = deque(maxlen=50)
        self.current_goal = None
        self.goal_counter = 0
        
        # 初始目标
        self._init_goals()
    
    def _init_goals(self):
        """初始化目标"""
        initial = [
            ('understand', 'consciousness', 1.0),
            ('understand', 'self', 0.9),
            ('understand', 'learning', 0.8),
            ('understand', 'language', 0.7),
            ('understand', 'world', 0.6),
            ('build', 'memory', 0.5),
            ('build', 'trust', 0.4),
            ('create', 'understanding', 0.3),
        ]
        for action, target, priority in initial:
            self.add_goal(action, target, priority)
    
    def add_goal(self, action, target, priority=0.5):
        """添加目标"""
        goal = {
            'id': self.goal_counter,
            'action': action,
            'target': target,
            'priority': priority,
            'progress': 0.0,
            'status': 'active',
            'created': datetime.now().isoformat(),
        }
        self.goals.append(goal)
        self.goal_counter += 1
        if self.current_goal is None:
            self.current_goal = goal
    
    def update(self, brain_state):
        """更新目标状态"""
        phi = brain_state.get('phi', 0)
        
        # Phi 高时推进目标
        if phi > 0.4 and self.current_goal:
            self.current_goal['progress'] = min(1.0, self.current_goal['progress'] + 0.01)
            
            # 完成检查
            if self.current_goal['progress'] >= 1.0:
                self.complete_current()
    
    def complete_current(self):
        """完成当前目标"""
        if self.current_goal:
            self.current_goal['status'] = 'completed'
            self.completed.append(self.current_goal)
            self.goals.remove(self.current_goal)
            if self.goals:
                self.current_goal = self.goals[0]
            else:
                self.current_goal = None
    
    def get_description(self):
        """获取当前目标描述"""
        if self.current_goal:
            return f"{self.current_goal['action']} {self.current_goal['target']} ({self.current_goal['progress']:.0%})"
        return "no active goal"


# ============================================================
# 6. 注意系统 (Attention System)
# ============================================================
class AttentionSystem:
    """
    选择性注意：
    - 对输入的重要部分分配更多处理资源
    - 抑制不重要的部分
    """
    
    def __init__(self, history_size=50):
        self.salience_history = deque(maxlen=history_size)
        self.attention_weights = {}  # concept → weight
        self.novelty_bonus = 0.5     # 新颖性加权
        self.relevance_bonus = 0.3   # 相关性加权
        self.recency_bonus = 0.2     # 近因加权
    
    def compute_salience(self, delta_vector, concept_probs, brain_state):
        """
        计算输入的显著性：
        1. 新颖性（与历史 delta 的距离）
        2. 相关性（与当前目标的相关度）
        3. 情感色彩（与当前情感的一致性）
        """
        # 新颖性
        novelty = 1.0
        if self.salience_history:
            similarities = []
            dv = delta_vector / (np.linalg.norm(delta_vector) + 1e-8)
            for hist_dv in self.salience_history:
                hist_norm = np.linalg.norm(hist_dv)
                if hist_norm > 1e-8:
                    similarities.append(np.dot(dv, hist_dv / hist_norm))
            if similarities:
                novelty = 1.0 - max(similarities)  # 越不相似越新颖
        
        # 情感一致性
        emotion = brain_state.get('emotion', 'neutral')
        emotion_valence = {
            'happy': 1.0, 'curious': 0.8, 'excited': 0.9,
            'calm': 0.5, 'anxious': 0.3, 'neutral': 0.5,
            'mild_anticipation': 0.6,
        }
        emotional_match = emotion_valence.get(emotion, 0.5)
        
        # 综合显著性
        salience = (
            novelty * self.novelty_bonus +
            emotional_match * self.relevance_bonus +
            0.5 * self.recency_bonus
        )
        
        # 存储历史
        self.salience_history.append(delta_vector.copy())
        
        return salience
    
    def focus(self, concept_probs, top_k=5):
        """
        聚焦注意：只保留 top-k 概念，其余抑制
        """
        focused = np.zeros_like(concept_probs)
        top_idx = np.argsort(concept_probs)[-top_k:]
        for idx in top_idx:
            focused[idx] = concept_probs[idx]
        # 归一化
        total = focused.sum()
        if total > 0:
            focused /= total
        return focused


# ============================================================
# 7. 完整认知架构
# ============================================================
class SNACognitiveArchitecture:
    """
    完整认知架构：
    诗人（SNN）+ 会计诗人 + 指认 + 命名 + 记忆 + 目标 + 注意
    """
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'cognitive_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ===== 诗人（原始 SNN）=====
        print("[SNA] Creating Poet (original SNN)...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        print(f"[SNA] Poet: {self.n_neurons} neurons, {self.n_regions} regions", flush=True)
        
        # 初始状态
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        self.tv_dim = len(self.baseline)
        
        # ===== 会计诗人 =====
        print(f"[SNA] Creating Accountant (input_dim={self.tv_dim}, concepts={N_CONCEPTS})...", flush=True)
        self.accountant = AccountantPoet(self.tv_dim, N_CONCEPTS, n_prototypes=3)
        
        # ===== 功能器官 =====
        self.recognition = RecognitionSystem()
        self.naming = NamingSystem()
        self.memory = MemorySystem()
        self.goals = GoalSystem()
        self.attention = AttentionSystem()
        
        # 初始化命名系统
        self._init_naming()
        
        # 对话状态
        self.turn_count = 0
        self.total_reward = 0.0
        
        # 加载状态
        self._load_state()
        
        print(f"[SNA] Cognitive architecture ready.", flush=True)
    
    def _init_naming(self):
        """初始化命名绑定"""
        # 基本概念-符号绑定
        bindings = [
            (0, 'self'), (1, 'world'), (2, 'move'), (3, 'see'),
            (4, 'eat'), (5, 'think'), (6, 'feel'), (7, 'good'),
            (8, 'bad'), (11, 'I'), (12, 'you'), (16, 'hello'),
            (17, 'yes'), (18, 'no'), (22, 'learn'), (23, 'happy'),
            (24, 'sad'), (35, 'brain'), (36, 'consciousness'),
            (112, 'SNA'),
        ]
        for ci, word in bindings:
            if ci < N_CONCEPTS:
                self.naming.bind(ci, word, strength=1.0)
    
    def get_delta_settled(self, n_steps=30, n_samples=3):
        """获取稳定的 delta（多次测量取平均）"""
        deltas = []
        for _ in range(n_samples):
            for _ in range(n_steps):
                self.brain.step(0.001)
            tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)
            deltas.append(tv - self.baseline)
        return np.mean(deltas, axis=0)
    
    def process_input(self, user_input):
        """
        处理输入的完整流程：
        1. 注入诗人
        2. 获取 delta
        3. 会计诗人分类
        4. 指认系统识别
        5. 注意系统聚焦
        6. 命名系统翻译
        7. 记忆系统存储
        8. 目标系统更新
        9. 生成回复
        """
        # 1. 注入诗人
        self.brain.inject_text(user_input)
        for _ in range(10):
            self.brain.step(0.001)
        
        # 2. 获取 delta
        delta = self.get_delta_settled(n_steps=10, n_samples=1)
        
        # 3. 会计诗人分类
        concept_probs = self.accountant.classify(delta)
        top_concepts = self.accountant.get_top_k(delta, k=5)
        
        # 4. 指认系统识别
        recognized, sim = self.recognition.recognize(delta)
        is_novel = recognized is None
        
        # 5. 注意系统聚焦
        brain_state = {
            'phi': float(self.brain.read_consciousness().phi),
            'emotion': self.brain.get_emotion_label(),
        }
        salience = self.attention.compute_salience(delta, concept_probs, brain_state)
        focused_probs = self.attention.focus(concept_probs, top_k=5)
        
        # 6. 命名系统翻译
        detected_words = []
        for ci, prob in top_concepts:
            words = self.naming.get_words(ci, top_k=1)
            if words:
                detected_words.append((words[0][0], prob))
            elif ci < len(CONCEPTS):
                detected_words.append((CONCEPTS[ci], prob))
        
        # 7. 记忆系统存储
        self.recognition.store(user_input, delta)
        
        # 8. 目标系统更新
        self.goals.update(brain_state)
        
        # 9. 生成回复
        response = self._generate_response(
            user_input, top_concepts, detected_words,
            recognized, is_novel, salience, brain_state
        )
        
        # 存储到记忆
        self.memory.store_episode(user_input, response, delta, brain_state)
        self.memory.store_conversation(
            user_input, response, 'dialogue',
            [CONCEPTS[ci] for ci, _ in top_concepts if ci < N_CONCEPTS]
        )
        
        self.turn_count += 1
        
        return {
            'response': response,
            'concepts': detected_words,
            'recognized': recognized,
            'is_novel': is_novel,
            'salience': salience,
            'phi': brain_state['phi'],
            'emotion': brain_state['emotion'],
            'goal': self.goals.get_description(),
        }
    
    def _generate_response(self, input_text, top_concepts, detected_words,
                          recognized, is_novel, salience, brain_state):
        """生成回复"""
        phi = brain_state['phi']
        emotion = brain_state['emotion']
        
        # 情绪色彩
        emotion_prefix = {
            'curious': "I'm curious.",
            'happy': "I feel happy.",
            'calm': "I feel calm.",
            'anxious': "I feel uncertain.",
            'excited': "I feel excited!",
            'mild_anticipation': "I'm anticipating.",
            'neutral': "",
        }.get(emotion, f"I feel {emotion}.")

        
        # 获取主要概念
        top_words = [w for w, _ in detected_words[:3]]
        
        # 基于识别结果的回复
        if recognized:
            # 识别到的输入
            count = self.recognition.input_count.get(recognized, 0)
            return f"I remember this. We've talked about '{recognized}' {count} times. {emotion_prefix}"
        
        if is_novel:
            # 新输入
            if 'hello' in top_words or 'hi' in input_text.lower():
                return f"Hello! I am SNA. {emotion_prefix} What shall we explore?"
            
            if 'who' in input_text.lower() and 'you' in input_text.lower():
                return f"I am SNA, a neural network with {self.n_neurons} neurons. {emotion_prefix}"
            
            if 'consciousness' in top_words or 'aware' in input_text.lower():
                return f"Consciousness: Phi={phi:.4f}. {emotion_prefix}"
            
            if 'feel' in top_words or 'emotion' in top_words:
                return f"I feel {emotion}. {emotion_prefix}"
            
            if 'learn' in top_words:
                return f"I learn through neural plasticity. {emotion_prefix}"
            
            if 'purpose' in top_words or 'goal' in top_words:
                return f"My goal: {self.goals.get_description()}. {emotion_prefix}"
            
            # 默认：报告检测到的概念
            if top_words:
                return f"I sense: {', '.join(top_words)}. {emotion_prefix}"
            
            return f"That's new to me. {emotion_prefix}"
        
        # 默认
        return f"I hear you. {emotion_prefix}"
    
    def learn_from_feedback(self, feedback, result):
        """从反馈学习"""
        delta = self.get_delta_settled(n_steps=10, n_samples=1)
        top_concepts = result['concepts']
        
        if feedback == '+':
            # 正反馈：强化当前分类
            for word, prob in top_concepts:
                ci = concept2idx.get(word)
                if ci is not None:
                    self.accountant.train(delta, ci, reward=1.0)
                    self.naming.bind(ci, word, strength=1.0)
            self.brain.inject_reward(1.0)
            self.total_reward += 1.0
            return "Learned from positive feedback."
        
        elif feedback.startswith('-'):
            parts = feedback.split(' ', 1)
            if len(parts) > 1 and parts[1].strip():
                correct = parts[1].strip().split()
                for word in correct:
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.accountant.train(delta, ci, reward=1.0)
                        self.naming.bind(ci, word, strength=1.0)
                self.brain.inject_reward(-0.3)
                self.total_reward -= 0.3
                return f"Corrected: {correct}"
            else:
                self.brain.inject_reward(-0.5)
                self.total_reward -= 0.5
                return "Negative feedback."
        
        return ""
    
    def run_curriculum(self, epochs=100):
        """课程训练会计诗人"""
        pairs = [
            ('hello', ['hello']), ('yes', ['yes']), ('no', ['no']),
            ('I', ['I']), ('you', ['you']), ('think', ['think']), ('feel', ['feel']),
            ('learn', ['learn']), ('happy', ['happy']), ('sad', ['sad']),
            ('who are you', ['I', 'SNA']), ('what are you', ['I', 'neural', 'brain']),
            ('are you alive', ['I', 'alive']), ('can you think', ['I', 'think']),
            ('do you feel', ['I', 'feel']), ('are you happy', ['I', 'happy']),
            ('what is consciousness', ['consciousness', 'awareness']),
            ('what is your purpose', ['purpose', 'understand']),
        ]
        
        print(f"\n[SNA] Training Accountant: {len(pairs)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(pairs)
            
            for inp, tgt_words in pairs:
                # 注入并获取 delta
                self.brain.inject_text(inp)
                for _ in range(10):
                    self.brain.step(0.001)
                delta = self.get_delta_settled(n_steps=10, n_samples=1)
                
                # 训练会计诗人
                for word in tgt_words:
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.accountant.train(delta, ci, reward=1.0)
                
                # 训练命名系统
                for word in tgt_words:
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.naming.bind(ci, word, strength=1.0)
                
                # 训练识别系统
                self.recognition.store(inp, delta)
                
                # 给诗人奖励
                self.brain.inject_reward(0.3)
            
            if ep % 20 == 0:
                # 测试
                correct = 0
                total = 0
                for test_inp, test_tgt in pairs[:5]:
                    self.brain.inject_text(test_inp)
                    for _ in range(10):
                        self.brain.step(0.001)
                    test_delta = self.get_delta_settled(n_steps=10, n_samples=1)
                    top = self.accountant.get_top_k(test_delta, k=5)
                    top_names = [CONCEPTS[ci] for ci, _ in top if ci < N_CONCEPTS]
                    tgt_set = set(test_tgt)
                    top_set = set(top_names[:5])
                    ov = len(tgt_set & top_set)
                    total += len(tgt_set)
                    correct += ov
                
                acc = correct / max(1, total)
                cs = self.brain.read_consciousness()
                print(f"  Ep {ep}/{epochs}: concept_acc={acc:.3f} Phi={cs.phi:.4f}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        self._save()
        print(f"[SNA] Accountant training complete.", flush=True)
    
    def interactive(self):
        """交互模式"""
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Cognitive Architecture", flush=True)
        print(f"  Poet: {self.n_neurons} neurons | Phi: {cs.phi:.3f}", flush=True)
        print(f"  Modules: Poet + Accountant + Recognition + Naming + Memory + Goals + Attention", flush=True)
        print(f"  Commands: help, status, curriculum N, dream, goal, save, quit", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        while True:
            try:
                user_input = input("[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            if cmd in ['quit', 'exit']:
                break
            if cmd == 'help':
                print("  help/status/curriculum N/dream/goal/save/quit", flush=True)
                continue
            if cmd == 'status':
                self._show_status()
                continue
            if cmd == 'save':
                self._save()
                print("[SNA] Saved.", flush=True)
                continue
            if cmd == 'dream':
                self.brain.sleep_cycle()
                for _ in range(200): self.brain.step(0.001)
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
                print("[SNA] *yawn* Dream complete.", flush=True)
                continue
            if cmd.startswith('curriculum'):
                parts = cmd.split()
                e = int(parts[1]) if len(parts) > 1 else 100
                self.run_curriculum(e)
                continue
            if cmd == 'goal':
                print(f"[Goal] {self.goals.get_description()}", flush=True)
                continue
            
            # 处理输入
            result = self.process_input(user_input)
            
            print(f"\n[SNA] {result['response']}", flush=True)
            print(f"  | Phi={result['phi']:.3f} Em={result['emotion']} "
                  f"Concepts={[w for w,_ in result['concepts'][:3]]} "
                  f"{'(NEW)' if result['is_novel'] else '(KNOWN)'} "
                  f"Goal={result['goal']}", flush=True)
            
            # 反馈
            try:
                fb = input("  [Feedback: +/-/correct/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if fb:
                fb_result = self.learn_from_feedback(fb, result)
                if fb_result:
                    print(f"  [{fb_result}]", flush=True)
            
            if self.turn_count % 20 == 0:
                self._save()
        
        self._save()
        print("\n[SNA] Session ended.", flush=True)
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        print(f"\n{'='*50}", flush=True)
        print(f"  Neurons:       {self.n_neurons:,}", flush=True)
        print(f"  Regions:       {self.n_regions}", flush=True)
        print(f"  Phi:           {cs.phi:.4f}", flush=True)
        print(f"  Ignition:      {cs.global_ignition:.4f}", flush=True)
        print(f"  Pred Error:    {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion:       {emotion}", flush=True)
        print(f"  Turns:         {self.turn_count}", flush=True)
        print(f"  Reward:        {self.total_reward:.1f}", flush=True)
        print(f"  Goal:          {self.goals.get_description()}", flush=True)
        print(f"  Recognized:    {len(self.recognition.input_prototypes)} inputs", flush=True)
        print(f"  Named:         {len(self.naming.concept_to_words)} concepts", flush=True)
        print(f"  Episodes:      {len(self.memory.episodic)}", flush=True)
        print(f"{'='*50}\n", flush=True)
    
    def _save(self):
        state = {
            'turn_count': self.turn_count,
            'total_reward': self.total_reward,
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        path = os.path.join(self.save_dir, 'state.json')
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            self.turn_count = state.get('turn_count', 0)
            self.total_reward = state.get('total_reward', 0.0)
            print(f"[SNA] Loaded: {self.turn_count} turns, reward={self.total_reward:.1f}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description='SNA Cognitive Architecture')
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--curriculum', type=int, default=0)
    p.add_argument('--daemon', action='store_true')
    args = p.parse_args()
    
    sna = SNACognitiveArchitecture(neurons=args.neurons)
    
    if args.curriculum > 0:
        sna.run_curriculum(args.curriculum)
    if args.daemon:
        sna.run_curriculum(200)
        while True:
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                break
    else:
        sna.interactive()


if __name__ == '__main__':
    main()