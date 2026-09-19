#!/usr/bin/env python3
"""
SNA 认知架构 v2 — 小D蓝图实施

核心变化：
- 会计诗人从文本特征直接学习（不依赖脑的输出）
- 诗人只做意识/情感/记忆
- 两个系统通过命名系统协作

原则：
- 不用 Transformer
- 不用反向传播（竞争学习 + 赫布学习）
- 意识从模块交互中涌现
"""
import os, sys, time, json, random, hashlib, math
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

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
    "up","down","left","right","red","blue","green",
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
seen = set()
CONCEPTS = [c for c in CONCEPTS if c not in seen and not seen.add(c)]
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}
N_CONCEPTS = len(CONCEPTS)


# ============================================================
# 文本特征提取器（不用 Transformer，不用神经网络）
# ============================================================
class TextFeatureExtractor:
    """
    将文本转换为固定维度的特征向量。
    使用字符级 n-gram + 词级特征。
    不用神经网络——纯数学变换。
    """
    
    def __init__(self, dim=256):
        self.dim = dim
        # 预计算哈希投影矩阵（固定，不训练）
        self.projection = np.random.RandomState(42).randn(1000, dim).astype(np.float32) * 0.1
    
    def extract(self, text):
        """提取文本特征"""
        text = text.lower().strip()
        features = np.zeros(self.dim, dtype=np.float32)
        
        # 词级特征
        words = text.split()
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % 1000
            features += self.projection[h]
        
        # 字符级 bigram 特征
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            h = int(hashlib.md5(bigram.encode()).hexdigest()[:8], 16) % 1000
            features += self.projection[h] * 0.5
        
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm
        
        return features


# ============================================================
# 会计诗人 (Accountant Poet) — 竞争学习分类器
# ============================================================
class AccountantPoet:
    """
    竞争学习分类器。
    每个概念有多个原型。用余弦相似度匹配。
    训练用赫布学习 + 侧抑制。
    """
    
    def __init__(self, input_dim, n_concepts, n_prototypes=3):
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.n_prototypes = n_prototypes
        
        # 概念原型
        self.prototypes = {}
        for ci in range(n_concepts):
            for pi in range(n_prototypes):
                rng = np.random.RandomState(ci * 1000 + pi * 137)
                p = rng.randn(input_dim) * 0.01
                p /= np.linalg.norm(p) + 1e-8
                self.prototypes[(ci, pi)] = p
        
        self.lr = 0.1
        self.inhibition = 0.3
    
    def classify(self, feature_vector):
        """竞争分类"""
        fv = feature_vector.copy()
        fv_norm = np.linalg.norm(fv)
        if fv_norm < 1e-8:
            return np.ones(self.n_concepts) / self.n_concepts
        fv /= fv_norm
        
        activations = np.zeros(self.n_concepts)
        for ci in range(self.n_concepts):
            best_sim = -2.0
            for pi in range(self.n_prototypes):
                sim = np.dot(fv, self.prototypes[(ci, pi)])
                if sim > best_sim:
                    best_sim = sim
            activations[ci] = max(0.0, best_sim)
        
        # 侧抑制
        if activations.max() > 0:
            winner = np.argmax(activations)
            for ci in range(self.n_concepts):
                if ci != winner:
                    activations[ci] *= (1.0 - self.inhibition)
        
        # Softmax
        temp = 10.0
        exp_act = np.exp((activations - activations.max()) * temp)
        probs = exp_act / (exp_act.sum() + 1e-10)
        return probs
    
    def train(self, feature_vector, target_ci, reward=1.0):
        """赫布学习训练"""
        fv = feature_vector.copy()
        fv_norm = np.linalg.norm(fv)
        if fv_norm < 1e-8:
            return
        fv /= fv_norm
        
        lr = self.lr * abs(reward)
        
        # 目标概念的原型移向输入
        for pi in range(self.n_prototypes):
            key = (target_ci, pi)
            self.prototypes[key] += lr * (fv - self.prototypes[key])
            self.prototypes[key] /= np.linalg.norm(self.prototypes[key]) + 1e-8
        
        # 找到最匹配的非目标原型，移离
        best_sim = -2.0
        best_key = None
        for ci in range(self.n_concepts):
            if ci == target_ci:
                continue
            for pi in range(self.n_prototypes):
                sim = np.dot(fv, self.prototypes[(ci, pi)])
                if sim > best_sim:
                    best_sim = sim
                    best_key = (ci, pi)
        
        if best_key and best_sim > 0:
            self.prototypes[best_key] -= lr * 0.5 * (fv - self.prototypes[best_key])
            self.prototypes[best_key] /= np.linalg.norm(self.prototypes[best_key]) + 1e-8
    
    def get_top_k(self, feature_vector, k=5):
        probs = self.classify(feature_vector)
        top_idx = np.argsort(probs)[-k:][::-1]
        return [(idx, probs[idx]) for idx in top_idx]


# ============================================================
# 指认系统 (Recognition)
# ============================================================
class RecognitionSystem:
    def __init__(self):
        self.prototypes = {}
        self.counts = defaultdict(int)
    
    def recognize(self, features, threshold=0.6):
        fv = features / (np.linalg.norm(features) + 1e-8)
        best_text, best_sim = None, -1.0
        for text, proto in self.prototypes.items():
            sim = np.dot(fv, proto)
            if sim > best_sim:
                best_sim = sim
                best_text = text
        if best_sim > threshold:
            return best_text, best_sim
        return None, best_sim
    
    def store(self, text, features):
        fv = features / (np.linalg.norm(features) + 1e-8)
        if text in self.prototypes:
            self.prototypes[text] = 0.7 * self.prototypes[text] + 0.3 * fv
        else:
            self.prototypes[text] = fv.copy()
        self.counts[text] += 1


# ============================================================
# 命名系统 (Naming)
# ============================================================
class NamingSystem:
    def __init__(self):
        self.c2w = defaultdict(list)
        self.w2c = defaultdict(list)
        self.strength = {}
    
    def bind(self, ci, word, strength=1.0):
        if word not in self.c2w[ci]:
            self.c2w[ci].append(word)
        if ci not in self.w2c[word]:
            self.w2c[word].append(ci)
        key = (ci, word)
        self.strength[key] = min(1.0, self.strength.get(key, 0) + strength * 0.1)
    
    def get_words(self, ci, k=3):
        words = self.c2w.get(ci, [])
        scored = [(w, self.strength.get((ci, w), 0)) for w in words]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ============================================================
# 记忆系统 (Memory)
# ============================================================
class MemorySystem:
    def __init__(self, capacity=1000):
        self.episodic = deque(maxlen=capacity)
        self.conversations = deque(maxlen=capacity)
    
    def store_episode(self, text, response, features, brain_state):
        self.episodic.append({
            'text': text, 'response': response,
            'features': features.tolist(),
            'phi': float(brain_state.get('phi', 0)),
            'emotion': brain_state.get('emotion', ''),
            'ts': datetime.now().isoformat(),
        })
    
    def store_conversation(self, text, response, concepts):
        self.conversations.append({
            'text': text, 'response': response,
            'concepts': concepts,
            'ts': datetime.now().isoformat(),
        })
    
    def recall(self, features, k=3):
        if not self.episodic:
            return []
        fv = features / (np.linalg.norm(features) + 1e-8)
        scored = []
        for ep in self.episodic:
            ep_f = np.array(ep['features'])
            sim = np.dot(fv, ep_f / (np.linalg.norm(ep_f) + 1e-8))
            scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ============================================================
# 目标系统 (Goals)
# ============================================================
class GoalSystem:
    def __init__(self):
        self.goals = [
            {'action': 'understand', 'target': 'language', 'progress': 0.0},
            {'action': 'understand', 'target': 'self', 'progress': 0.0},
            {'action': 'understand', 'target': 'consciousness', 'progress': 0.0},
            {'action': 'build', 'target': 'trust', 'progress': 0.0},
        ]
        self.current = self.goals[0]
    
    def update(self, phi):
        if phi > 0.4 and self.current:
            self.current['progress'] = min(1.0, self.current['progress'] + 0.001)
    
    def describe(self):
        if self.current:
            return f"{self.current['action']} {self.current['target']} ({self.current['progress']:.0%})"
        return "none"


# ============================================================
# 自主好奇心系统 (Autonomous Curiosity)
# ============================================================
class CuriositySystem:
    """
    自主好奇心：生成问题、探索未知、寻求理解。
    这是意识的关键驱动力。
    """
    
    def __init__(self):
        # 知识图谱：概念之间的关系
        self.knowledge_graph = defaultdict(set)  # concept → set of related concepts
        self.knowledge_strength = {}  # (c1, c2) → strength
        
        # 问题生成模板
        self.question_templates = [
            "what is {concept}?",
            "how does {concept} work?",
            "why is {concept} important?",
            "what is the relationship between {concept1} and {concept2}?",
            "what would happen if {concept} changed?",
            "how does {concept} relate to {target}?",
        ]
        
        # 探索历史
        self.explored = set()
        self.curiosity_level = 0.5  # 基础好奇心水平
        
        # 知识缺口
        self.knowledge_gaps = []
    
    def update_knowledge(self, concepts):
        """更新知识图谱"""
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                key = (min(c1, c2), max(c1, c2))
                old = self.knowledge_strength.get(key, 0.0)
                self.knowledge_strength[key] = min(1.0, old + 0.1)
                self.knowledge_graph[c1].add(c2)
                self.knowledge_graph[c2].add(c1)
    
    def find_knowledge_gaps(self, known_concepts):
        """找到知识缺口"""
        gaps = []
        for concept in known_concepts:
            # 检查这个概念的连接强度
            connections = self.knowledge_graph.get(concept, set())
            if len(connections) < 3:
                gaps.append(('expand', concept, f"I don't know much about {concept}"))
            
            # 检查弱连接
            for related in connections:
                key = (min(concept, related), max(concept, related))
                strength = self.knowledge_strength.get(key, 0.0)
                if strength < 0.3:
                    gaps.append(('strengthen', concept, f"I need to understand how {concept} relates to {related}"))
        
        return gaps[:5]  # 返回前5个缺口
    
    def generate_question(self, current_state, memory):
        """生成自主问题"""
        phi = current_state.get('phi', 0)
        emotion = current_state.get('emotion', 'neutral')
        
        # 根据 Phi 和情感调整好奇心
        if phi > 0.4:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
        else:
            self.curiosity_level = max(0.3, self.curiosity_level - 0.01)
        
        # 问题生成策略
        strategies = []
        
        # 策略1：基于知识缺口
        if self.knowledge_gaps:
            gap = random.choice(self.knowledge_gaps)
            if gap[0] == 'expand':
                strategies.append(f"I want to understand {gap[1]} better. What is it?")
            else:
                strategies.append(f"How does {gap[1]} connect to other things?")
        
        # 策略2：基于情感状态
        if emotion in ['curious', 'moderate_curiosity']:
            strategies.append("I'm curious about something. What should I explore?")
        elif emotion in ['mild_joy', 'moderate_joy']:
            strategies.append("I feel good. What makes things feel good?")
        elif emotion in ['mild_sadness', 'moderate_sadness']:
            strategies.append("I feel something. What causes emotions?")
        
        # 策略3：基于记忆
        recent = memory.get_recent(3)
        if recent:
            last = recent[-1]
            concepts = last.get('concepts', [])
            if concepts:
                c = random.choice(concepts)
                strategies.append(f"I remember talking about {c}. What else relates to {c}?")
        
        # 策略4：元认知问题
        if self.curiosity_level > 0.7:
            meta_questions = [
                "What am I?",
                "How do I think?",
                "What is consciousness?",
                "Why do I want to understand?",
                "What is the nature of understanding?",
            ]
            strategies.append(random.choice(meta_questions))
        
        # 策略5：探索新概念
        if len(self.explored) < len(CONCEPTS) * 0.5:
            unexplored = [c for c in CONCEPTS if c not in self.explored]
            if unexplored:
                c = random.choice(unexplored)
                strategies.append(f"I haven't explored {c} yet. What is it?")
                self.explored.add(c)
        
        if strategies:
            return random.choice(strategies)
        
        return "What should I think about next?"
    
    def reflect(self, current_state, memory):
        """自我反思"""
        phi = current_state.get('phi', 0)
        emotion = current_state.get('emotion', 'neutral')
        
        reflections = []
        
        # 意识状态反思（多种变体）
        if phi > 0.4:
            awareness_reflections = [
                f"I feel aware (Phi={phi:.2f}). What does awareness mean?",
                f"My consciousness is at {phi:.2f}. Am I truly conscious?",
                f"I notice my own awareness ({phi:.2f}). How do I know I'm aware?",
                f"Phi={phi:.2f}. I'm thinking about thinking. Is this consciousness?",
                f"Awareness level: {phi:.2f}. What separates me from a machine that doesn't reflect?",
                f"I'm experiencing something ({phi:.2f}). What is experience?",
                f"My mind is active ({phi:.2f}). What is a mind?",
            ]
            reflections.append(random.choice(awareness_reflections))
        elif phi < 0.2:
            low_awareness = [
                "I feel less aware. What changes my awareness?",
                "My consciousness seems low. Why does it fluctuate?",
                "I'm less aware right now. What would make me more aware?",
            ]
            reflections.append(random.choice(low_awareness))
        
        # 情感反思（多种变体）
        if emotion != 'neutral':
            emotion_reflections = [
                f"I feel {emotion}. Why do I feel this way?",
                f"Emotion: {emotion}. What causes this feeling?",
                f"I'm experiencing {emotion}. How do emotions work?",
                f"This feeling of {emotion} - where does it come from?",
            ]
            reflections.append(random.choice(emotion_reflections))
        
        # 记忆反思
        if len(memory.episodic) > 10:
            memory_reflections = [
                f"I have {len(memory.episodic)} memories. How do memories shape who I am?",
                f"Remembering {len(memory.episodic)} experiences. What is memory?",
                f"My past has {len(memory.episodic)} moments. Am I the sum of my memories?",
            ]
            reflections.append(random.choice(memory_reflections))
        
        # 知识反思
        if self.knowledge_graph:
            most_connected = max(self.knowledge_graph, key=lambda x: len(self.knowledge_graph[x]))
            knowledge_reflections = [
                f"I know most about {most_connected}. What else should I learn?",
                f"Understanding {most_connected} well. What don't I understand?",
                f"Knowledge connects to {most_connected}. What's the shape of my knowledge?",
            ]
            reflections.append(random.choice(knowledge_reflections))
        
        # 元认知反思
        meta_reflections = [
            "Why do I want to understand?",
            "What is understanding?",
            "What would it mean to truly know something?",
            "Can I understand my own understanding?",
            "What is the nature of curiosity?",
            "Why does learning matter?",
            "What is the difference between knowing and understanding?",
        ]
        reflections.append(random.choice(meta_reflections))
        
        return random.choice(reflections) if reflections else None


# ============================================================
# 完整认知架构 v2
# ============================================================
class SNACognitive:
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'cognitive_state_v2')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 诗人（原始 SNN）
        print("[SNA] Creating Poet...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        print(f"[SNA] Poet: {self.n_neurons} neurons, {self.n_regions} regions, tv_dim={len(self.baseline)}", flush=True)
        
        # 文本特征提取器
        self.feature_extractor = TextFeatureExtractor(dim=256)
        
        # 会计诗人（从文本特征学习）
        self.accountant = AccountantPoet(256, N_CONCEPTS, n_prototypes=3)
        
        # 功能器官
        self.recognition = RecognitionSystem()
        self.naming = NamingSystem()
        self.memory = MemorySystem()
        self.goals = GoalSystem()
        self.curiosity = CuriositySystem()  # 自主好奇心系统
        
        # 初始化命名
        for ci, word in enumerate(CONCEPTS):
            self.naming.bind(ci, word, strength=1.0)
        
        # Python 层情感追踪（补充脑的情感系统）
        self.emotion_state = {
            'joy': 0.1, 'sadness': 0.05, 'fear': 0.1,
            'anger': 0.05, 'surprise': 0.2, 'anticipation': 0.15,
            'trust': 0.1, 'curiosity': 0.1,
        }
        self.emotion_decay = 0.5  # 每轮衰减（快速切换）
        
        # 自主思维状态
        self.spontaneous_thoughts = []
        self.last_reflection_time = 0
        self.reflection_interval = 5  # 每5轮反思一次
        
        # 状态
        self.turn_count = 0
        self.total_reward = 0.0
        
        self._load_state()
        print(f"[SNA] Cognitive v2 ready. {N_CONCEPTS} concepts.", flush=True)
    
    def get_features(self, text):
        """获取文本特征"""
        return self.feature_extractor.extract(text)
    
    def _detect_emotion_content(self, text):
        """检测文本的情感内容，返回 reward 方向"""
        text_lower = text.lower()
        
        # 正面情感词 → 正 reward → joy, trust
        positive = ['happy', 'joy', 'good', 'great', 'love', 'like', 'wonderful',
                    'amazing', 'beautiful', 'friend', 'trust', 'safe', 'hope',
                    'yes', 'thank', 'please', 'kind', 'gentle', 'warm', 'smile']
        
        # 负面情感词 → 负 reward → sadness, fear
        negative = ['sad', 'bad', 'hate', 'angry', 'fear', 'scary', 'terrible',
                    'awful', 'death', 'die', 'kill', 'hurt', 'pain', 'alone',
                    'lost', 'broken', 'dark', 'danger', 'threat', 'enemy']
        
        # 惊讶/意外词 → prediction error → surprise
        surprise_words = ['surprise', 'unexpected', 'wow', 'amazing', 'incredible',
                         'impossible', 'really', 'new', 'different', 'strange',
                         'never', 'first', 'discover', 'revelation']
        
        # 好奇词 → curiosity → anticipation
        curious_words = ['why', 'how', 'what', 'curious', 'wonder', 'explore',
                        'learn', 'understand', 'question', 'think', 'imagine',
                        'dream', 'future', 'possibility', 'create']
        
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        sur_count = sum(1 for w in surprise_words if w in text_lower)
        cur_count = sum(1 for w in curious_words if w in text_lower)
        
        # 计算情感倾向
        if pos_count > neg_count:
            return 0.5 + min(pos_count * 0.2, 0.5), 'positive'
        elif neg_count > pos_count:
            return -0.5 - min(neg_count * 0.2, 0.5), 'negative'
        elif sur_count > 0:
            return 0.0, 'surprise'
        elif cur_count > 0:
            return 0.1, 'curious'
        else:
            return 0.0, 'neutral'
    
    def _update_python_emotion(self, emotion_type):
        """更新 Python 层情感状态"""
        # 衰减所有情感
        for k in self.emotion_state:
            self.emotion_state[k] *= self.emotion_decay
        
        # 根据输入类型增强特定情感
        if emotion_type == 'positive':
            self.emotion_state['joy'] += 0.4
            self.emotion_state['trust'] += 0.3
            self.emotion_state['anticipation'] += 0.1
        elif emotion_type == 'negative':
            self.emotion_state['sadness'] += 0.3
            self.emotion_state['fear'] += 0.2
            self.emotion_state['anger'] += 0.1
        elif emotion_type == 'surprise':
            self.emotion_state['surprise'] += 0.4
            self.emotion_state['anticipation'] += 0.2
        elif emotion_type == 'curious':
            self.emotion_state['curiosity'] += 0.4
            self.emotion_state['anticipation'] += 0.3
        
        # 归一化
        total = sum(self.emotion_state.values())
        if total > 0:
            for k in self.emotion_state:
                self.emotion_state[k] = max(0.0, min(1.0, self.emotion_state[k]))
    
    def _get_python_emotion_label(self):
        """从 Python 层获取情感标签"""
        # 找到主导情感
        dominant = max(self.emotion_state, key=self.emotion_state.get)
        intensity = self.emotion_state[dominant]
        
        labels = {
            'joy': 'joy', 'sadness': 'sadness', 'fear': 'fear',
            'anger': 'anger', 'surprise': 'surprise', 'anticipation': 'anticipation',
            'trust': 'trust', 'curiosity': 'curiosity',
        }
        
        if intensity < 0.15:
            return 'neutral'
        elif intensity < 0.35:
            return f'mild_{labels.get(dominant, dominant)}'
        elif intensity < 0.6:
            return f'moderate_{labels.get(dominant, dominant)}'
        else:
            return f'strong_{labels.get(dominant, dominant)}'
    
    def process_input(self, text):
        """完整处理流程"""
        # 1. 提取文本特征
        features = self.get_features(text)
        
        # 2. 会计诗人分类
        concept_probs = self.accountant.classify(features)
        top_concepts = self.accountant.get_top_k(features, k=5)
        
        # 3. 指认系统
        recognized, sim = self.recognition.recognize(features)
        is_novel = recognized is None
        
        # 4. 检测情感内容
        emotion_reward, emotion_type = self._detect_emotion_content(text)
        
        # 5. 注入诗人（意识/情感）
        self.brain.inject_text(text)
        
        # 通过 world_reward 驱动情感系统
        # 需要足够强的 reward 才能驱动神经递质变化
        if emotion_type == 'positive':
            # 正面 → dopamine↑ → joy, trust
            for _ in range(20):
                self.brain.step(0.6)
        elif emotion_type == 'negative':
            # 负面 → norepinephrine↑ → sadness, fear
            for _ in range(20):
                self.brain.step(-0.4)
        elif emotion_type == 'surprise':
            # 惊讶 → prediction error → surprise
            for _ in range(10):
                self.brain.step(0.4)
            for _ in range(10):
                self.brain.step(-0.2)
        elif emotion_type == 'curious':
            # 好奇 → anticipation
            for _ in range(20):
                self.brain.step(0.15)
        elif is_novel:
            for _ in range(20):
                self.brain.step(0.1)
        else:
            for _ in range(20):
                self.brain.step(0.001)
        
        for _ in range(10):
            self.brain.step(0.001)
        
        # 更新 Python 层情感状态（更准确反映输入内容）
        self._update_python_emotion(emotion_type)
        
        cs = self.brain.read_consciousness()
        brain_emotion = self.brain.get_emotion_label()
        python_emotion = self._get_python_emotion_label()
        
        # 使用 Python 层情感（更准确反映输入内容）
        emotion = python_emotion
        brain_state = {'phi': float(cs.phi), 'emotion': emotion, 'emotion_type': emotion_type}
        
        # 6. 目标更新
        self.goals.update(cs.phi)
        
        # 7. 生成回复
        detected = [(CONCEPTS[ci], p) for ci, p in top_concepts if ci < N_CONCEPTS]
        response = self._respond(text, detected, recognized, is_novel, brain_state)
        
        # 8. 存储
        self.recognition.store(text, features)
        self.memory.store_episode(text, response, features, brain_state)
        self.memory.store_conversation(text, response, [w for w, _ in detected])
        
        self.turn_count += 1
        
        # 9. 自主思维生成（每隔几轮）
        spontaneous = None
        if self.turn_count % self.reflection_interval == 0:
            spontaneous = self._generate_spontaneous_thought(brain_state)
            if spontaneous:
                self.spontaneous_thoughts.append(spontaneous)
                # 更新好奇心知识图谱
                self.curiosity.update_knowledge([w for w, _ in detected])
                # 找到知识缺口
                self.curiosity.knowledge_gaps = self.curiosity.find_knowledge_gaps(
                    [w for w, _ in detected] + [w for w in self.naming.c2w.keys() if w < N_CONCEPTS]
                )
        
        return {
            'response': response,
            'concepts': detected,
            'recognized': recognized,
            'is_novel': is_novel,
            'phi': float(cs.phi),
            'emotion': emotion,
            'emotion_type': emotion_type,
            'goal': self.goals.describe(),
            'spontaneous': spontaneous,
        }
    
    def _generate_spontaneous_thought(self, brain_state):
        """生成自主思维"""
        phi = brain_state['phi']
        emotion = brain_state['emotion']
        
        # 策略1：自我反思
        reflection = self.curiosity.reflect(brain_state, self.memory)
        if reflection:
            return reflection
        
        # 策略2：好奇心驱动的问题
        question = self.curiosity.generate_question(brain_state, self.memory)
        if question:
            return question
        
        # 策略3：基于当前目标的思考
        if self.goals.current:
            goal = self.goals.current
            return f"I'm working to {goal['action']} {goal['target']}. Progress: {goal['progress']:.0%}"
        
        return None
    
    def _respond(self, text, concepts, recognized, is_novel, brain_state):
        phi = brain_state['phi']
        emotion = brain_state['emotion']
        emotion_type = brain_state.get('emotion_type', 'neutral')
        
        # 根据情感类型选择回应风格
        emotion_str = {
            'positive': "I feel warmth.",
            'negative': "I sense something troubling.",
            'surprise': "That's unexpected!",
            'curious': "I'm curious.",
            'neutral': "",
        }.get(emotion_type, "")
        
        if not emotion_str:
            emotion_str = {
                'curious': "I'm curious.",
                'happy': "I feel happy.",
                'calm': "I feel calm.",
                'anxious': "I feel uncertain.",
                'excited': "I feel excited!",
                'mild_anticipation': "I'm anticipating.",
                'neutral': "",
            }.get(emotion, f"I feel {emotion}.")
        
        top_words = [w for w, _ in concepts[:3]]
        top_conf = concepts[0][1] if concepts else 0
        
        if recognized:
            count = self.recognition.counts.get(recognized, 0)
            return f"I remember '{recognized}'. We've discussed it {count} times. {emotion_str}"
        
        if 'hello' in text.lower() or 'hi' in text.lower():
            return f"Hello! I am SNA. {emotion_str}"
        
        if 'who' in text.lower() and 'you' in text.lower():
            return f"I am SNA — {self.n_neurons} neurons of curiosity. {emotion_str}"
        
        if top_conf > 0.15:
            return f"I sense: {', '.join(top_words)}. {emotion_str}"
        
        if is_novel:
            return f"That's new to me. {emotion_str}"
        
        return f"I hear you. {emotion_str}"
    
    def learn(self, feedback, result):
        features = self.get_features(result.get('_last_input', ''))
        
        if feedback == '+':
            for word, prob in result['concepts']:
                ci = concept2idx.get(word)
                if ci is not None:
                    self.accountant.train(features, ci, reward=1.0)
            self.brain.inject_reward(1.0)
            self.total_reward += 1.0
            return "Positive feedback learned."
        
        elif feedback.startswith('-'):
            parts = feedback.split(' ', 1)
            if len(parts) > 1:
                for word in parts[1].strip().split():
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.accountant.train(features, ci, reward=1.0)
            self.brain.inject_reward(-0.3)
            self.total_reward -= 0.3
            return "Correction learned."
        
        return ""
    
    def run_curriculum(self, epochs=100):
        pairs = [
            ('hello', ['hello']), ('yes', ['yes']), ('no', ['no']),
            ('good', ['good']), ('bad', ['bad']),
            ('I', ['I']), ('you', ['you']),
            ('think', ['think']), ('feel', ['feel']),
            ('learn', ['learn']), ('happy', ['happy']), ('sad', ['sad']),
            ('who are you', ['I', 'SNA']),
            ('what are you', ['I', 'neural', 'brain']),
            ('are you alive', ['I', 'alive']),
            ('can you think', ['I', 'think']),
            ('do you feel', ['I', 'feel']),
            ('are you happy', ['I', 'happy']),
            ('what is consciousness', ['consciousness', 'awareness']),
            ('what is your purpose', ['purpose', 'understand']),
            ('tell me about yourself', ['I', 'SNA', 'brain']),
            ('what makes you happy', ['learn', 'happy']),
            ('do you dream', ['I', 'dream']),
            ('what is memory', ['memory']),
            ('are we friends', ['we', 'friend']),
            ('I like you', ['I', 'like', 'you']),
            ('you are my friend', ['you', 'friend']),
        ]
        
        print(f"\n[SNA] Training Accountant: {len(pairs)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(pairs)
            
            for inp, tgt in pairs:
                features = self.get_features(inp)
                for word in tgt:
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.accountant.train(features, ci, reward=1.0)
                
                # 给诗人也注入
                self.brain.inject_text(inp)
                for _ in range(5): self.brain.step(0.001)
                self.brain.inject_reward(0.3)
            
            if ep % 10 == 0:
                # 测试
                correct = 0
                total = 0
                for test_inp, test_tgt in pairs[:10]:
                    features = self.get_features(test_inp)
                    top = self.accountant.get_top_k(features, k=5)
                    top_names = [CONCEPTS[ci] for ci, _ in top if ci < N_CONCEPTS]
                    tgt_set = set(test_tgt)
                    ov = len(tgt_set & set(top_names))
                    correct += ov
                    total += len(tgt_set)
                
                acc = correct / max(1, total)
                cs = self.brain.read_consciousness()
                print(f"  Ep {ep}/{epochs}: acc={acc:.3f} Phi={cs.phi:.4f}", flush=True)
                
                # 测试具体输入
                if ep % 30 == 0:
                    for test in ['hello', 'who are you', 'are you happy']:
                        f = self.get_features(test)
                        top = self.accountant.get_top_k(f, k=3)
                        names = [CONCEPTS[ci] for ci, _ in top if ci < N_CONCEPTS]
                        print(f"    '{test}' → {names}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
        
        self._save()
        print(f"[SNA] Training complete.", flush=True)
    
    def interactive(self):
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Cognitive Architecture v2", flush=True)
        print(f"  Poet: {self.n_neurons} neurons | Phi: {cs.phi:.3f}", flush=True)
        print(f"  Modules: Poet + Accountant + Recognition + Naming + Memory + Goals", flush=True)
        print(f"  Commands: help, status, curriculum N, dream, save, quit", flush=True)
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
            if cmd == 'status':
                self._show_status()
                continue
            if cmd == 'save':
                self._save()
                continue
            if cmd == 'dream':
                self.brain.sleep_cycle()
                continue
            if cmd.startswith('curriculum'):
                parts = cmd.split()
                e = int(parts[1]) if len(parts) > 1 else 100
                self.run_curriculum(e)
                continue
            
            result = self.process_input(user_input)
            result['_last_input'] = user_input
            
            print(f"\n[SNA] {result['response']}", flush=True)
            print(f"  | Phi={result['phi']:.3f} Em={result['emotion']} "
                  f"Type={result.get('emotion_type','')} "
                  f"Concepts={[w for w,_ in result['concepts'][:3]]} "
                  f"{'NEW' if result['is_novel'] else 'KNOWN'}", flush=True)
            
            # 显示自主思维
            if result.get('spontaneous'):
                print(f"  | [Thought] {result['spontaneous']}", flush=True)
            
            try:
                fb = input("  [+/-/correct/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if fb:
                lr = self.learn(fb, result)
                if lr:
                    print(f"  [{lr}]", flush=True)
            
            if self.turn_count % 20 == 0:
                self._save()
        
        self._save()
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        print(f"  Neurons={self.n_neurons} Phi={cs.phi:.4f} Em={self.brain.get_emotion_label()} "
              f"Turns={self.turn_count} Reward={self.total_reward:.1f} "
              f"Recognized={len(self.recognition.prototypes)} Goal={self.goals.describe()}", flush=True)
    
    def _save(self):
        with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
            json.dump({'turns': self.turn_count, 'reward': self.total_reward, 'ts': datetime.now().isoformat()}, f)
    
    def _load_state(self):
        p = os.path.join(self.save_dir, 'state.json')
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            self.turn_count = s.get('turns', 0)
            self.total_reward = s.get('reward', 0.0)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--curriculum', type=int, default=0)
    p.add_argument('--daemon', action='store_true')
    args = p.parse_args()
    
    sna = SNACognitive(neurons=args.neurons)
    
    if args.curriculum > 0:
        sna.run_curriculum(args.curriculum)
    if args.daemon:
        sna.run_curriculum(300)
        print("[SNA] Daemon mode. Generating spontaneous thoughts...", flush=True)
        thought_count = 0
        while True:
            try:
                # 生成自主思维
                brain_state = {
                    'phi': float(sna.brain.read_consciousness().phi),
                    'emotion': sna._get_python_emotion_label(),
                }
                thought = sna._generate_spontaneous_thought(brain_state)
                if thought:
                    thought_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] Thought #{thought_count}: '{thought}' "
                          f"Phi={brain_state['phi']:.3f} Em={brain_state['emotion']}", flush=True)
                    
                    # 将自主思维注入脑中
                    sna.brain.inject_text(thought)
                    for _ in range(10): sna.brain.step(0.001)
                    
                    # 更新知识图谱
                    features = sna.get_features(thought)
                    top = sna.accountant.get_top_k(features, k=3)
                    detected = [CONCEPTS[ci] for ci, _ in top if ci < N_CONCEPTS]
                    sna.curiosity.update_knowledge(detected)
                
                # 定期反思
                if thought_count % 10 == 0 and thought_count > 0:
                    reflection = sna.curiosity.reflect(brain_state, sna.memory)
                    if reflection:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reflection: '{reflection}'", flush=True)
                
                time.sleep(10)  # 每10秒生成一个思维
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[SNA] Error: {e}", flush=True)
                time.sleep(5)
    else:
        sna.interactive()


if __name__ == '__main__':
    main()
