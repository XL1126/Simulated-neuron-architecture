#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
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
import psutil
from seed import Seed, GenomeConstants, ConceptNode
from growth_monitor import GrowthMonitor


# ============================================================
# 语义理解系统 — 真正理解语言
# ============================================================
class SemanticUnderstanding:
    """
    语义理解：不只是检测关键词，而是理解句子的含义。

    核心机制：
    1. 词性标注 — 识别主语、谓语、宾语
    2. 语义角色 — 谁对谁做了什么
    3. 概念关联 — 用概念树理解词与词的关系
    4. 上下文记忆 — 记住对话内容并使用
    """

    def __init__(self, concept_tree):
        self.concept_tree = concept_tree
        self.word_cache = {}  # 词性缓存

        # 词性分类
        self.subjects = {'我', '你', '他', '她', '它', '我们', '他们', '大家', '人类', 'AI'}
        self.verbs = {'是', '有', '在', '想', '看', '听', '说', '做', '走', '来', '去', '喜欢', '讨厌', '知道', '理解', '感觉', '希望', '需要'}
        self.adjectives = {'好', '坏', '大', '小', '快', '慢', '热', '冷', '开心', '难过', '漂亮', '聪明', '有趣', '无聊'}
        self.time_words = {'今天', '昨天', '明天', '现在', '刚才', '以后', '以前', '早上', '晚上'}
        self.place_words = {'这里', '那里', '家里', '学校', '公司', '外面', '里面'}
        self.question_words = {'什么', '怎么', '为什么', '哪里', '谁', '几', '多少', '吗', '呢'}
        self.emotion_words = {'开心', '难过', '生气', '害怕', '惊讶', '讨厌', '喜欢', '爱', '恨', '担心', '紧张', '放松'}

    def understand(self, text, existing_concepts):
        """
        理解一句话的含义。

        返回: {
            'subject': 主语,
            'verb': 谓语,
            'object': 宾语,
            'time': 时间,
            'place': 地点,
            'emotion': 情感,
            'is_question': 是否问句,
            'question_type': 问题类型,
            'concepts': 涉及的概念,
            'meaning': 语义摘要,
        }
        """
        import re
        # 清洗文本
        clean = re.sub(r'[，。！？、；：""''（）《》\s]', '', text)
        if len(clean) < 2:
            return {'meaning': '空', 'concepts': []}

        # 1. 词性标注
        words = self._segment(clean)
        tagged = self._tag_pos(words)

        # 2. 提取语义角色
        subject = self._find_subject(tagged)
        verb = self._find_verb(tagged)
        obj = self._find_object(tagged)
        time = self._find_time(tagged)
        place = self._find_place(tagged)
        emotion = self._find_emotion(tagged)

        # 3. 判断是否问句
        is_question = any(q in text for q in ['？', '?', '吗', '呢', '什么', '怎么', '为什么', '哪里', '谁'])
        question_type = self._classify_question(text, tagged)

        # 4. 关联概念树
        concepts = self._link_to_concepts(tagged, existing_concepts)

        # 5. 生成语义摘要
        meaning = self._generate_meaning(subject, verb, obj, time, place, emotion, is_question)

        return {
            'subject': subject,
            'verb': verb,
            'object': obj,
            'time': time,
            'place': place,
            'emotion': emotion,
            'is_question': is_question,
            'question_type': question_type,
            'concepts': concepts,
            'meaning': meaning,
        }

    def _segment(self, text):
        """简单分词（基于词典）"""
        words = []
        i = 0
        while i < len(text):
            # 尝试最长匹配（4字）
            matched = False
            for length in [4, 3, 2]:
                if i + length <= len(text):
                    word = text[i:i+length]
                    if self._is_word(word):
                        words.append(word)
                        i += length
                        matched = True
                        break
            if not matched:
                words.append(text[i])
                i += 1
        return words

    def _is_word(self, text):
        """判断是否是词"""
        # 词典（大幅扩展）
        dictionary = {
            # 时间
            '今天', '昨天', '明天', '现在', '刚才', '以后', '以前',
            '早上', '晚上', '中午', '下午', '春天', '夏天', '秋天', '冬天',
            # 天气
            '天气', '下雨', '下雪', '晴天', '阴天', '刮风', '热', '冷',
            # 食物
            '苹果', '香蕉', '橘子', '西瓜', '葡萄', '食物', '水', '饭',
            # 活动
            '学习', '工作', '休息', '睡觉', '吃饭', '喝水', '看书', '听音乐',
            '跑步', '走路', '游泳', '唱歌', '跳舞', '画画', '写字',
            # 情感
            '开心', '难过', '生气', '害怕', '惊讶', '讨厌', '喜欢',
            '爱', '恨', '担心', '紧张', '放松', '快乐', '悲伤',
            # 描述
            '漂亮', '聪明', '有趣', '无聊', '好看', '好听', '好吃',
            '大', '小', '快', '慢', '热', '冷', '高', '低',
            # 物品
            '电脑', '手机', '网络', '书', '电影', '音乐', '颜色',
            # 抽象
            '自由', '快乐', '幸福', '成功', '失败', '生命', '死亡',
            '时间', '空间', '世界', '宇宙', '人类', '动物',
            # 关系
            '朋友', '家人', '父母', '孩子', '老师', '学生',
            # 动作
            '知道', '理解', '明白', '记得', '忘记', '希望', '需要',
            '想要', '可以', '能够', '应该', '必须',
            # 连接词
            '因为', '所以', '但是', '如果', '虽然', '而且', '或者',
            # 疑问词
            '什么', '怎么', '为什么', '哪里', '谁', '几', '多少',
        }
        return text in dictionary or len(text) == 1

    def _tag_pos(self, words):
        """词性标注"""
        tagged = []
        for word in words:
            if word in self.subjects:
                tagged.append((word, 'SUBJ'))
            elif word in self.verbs:
                tagged.append((word, 'VERB'))
            elif word in self.adjectives:
                tagged.append((word, 'ADJ'))
            elif word in self.time_words:
                tagged.append((word, 'TIME'))
            elif word in self.place_words:
                tagged.append((word, 'PLACE'))
            elif word in self.question_words:
                tagged.append((word, 'QUES'))
            elif word in self.emotion_words:
                tagged.append((word, 'EMOT'))
            else:
                tagged.append((word, 'NOUN'))
        return tagged

    def _find_subject(self, tagged):
        """找主语"""
        for word, pos in tagged:
            if pos == 'SUBJ':
                return word
        return None

    def _find_verb(self, tagged):
        """找谓语"""
        for word, pos in tagged:
            if pos == 'VERB':
                return word
        return None

    def _find_object(self, tagged):
        """找宾语"""
        # 宾语通常在谓语后面
        found_verb = False
        for word, pos in tagged:
            if pos == 'VERB':
                found_verb = True
            elif found_verb and pos in ('NOUN', 'ADJ', 'EMOT'):
                return word
        return None

    def _find_time(self, tagged):
        """找时间"""
        for word, pos in tagged:
            if pos == 'TIME':
                return word
        return None

    def _find_place(self, tagged):
        """找地点"""
        for word, pos in tagged:
            if pos == 'PLACE':
                return word
        return None

    def _find_emotion(self, tagged):
        """找情感词"""
        for word, pos in tagged:
            if pos == 'EMOT':
                return word
        return None

    def _classify_question(self, text, tagged):
        """分类问题类型"""
        if '什么' in text:
            return 'what'
        elif '怎么' in text:
            return 'how'
        elif '为什么' in text:
            return 'why'
        elif '哪里' in text or '哪儿' in text:
            return 'where'
        elif '谁' in text:
            return 'who'
        elif '几' in text or '多少' in text:
            return 'quantity'
        elif text.endswith('吗') or text.endswith('？'):
            return 'yesno'
        return None

    def _link_to_concepts(self, tagged, existing_concepts):
        """关联到概念树"""
        concepts = []
        for word, pos in tagged:
            # 直接匹配
            if word in existing_concepts:
                concepts.append(word)
            # 部分匹配
            else:
                for concept in existing_concepts:
                    if word in concept or concept in word:
                        concepts.append(concept)
                        break
        return list(set(concepts))

    def _generate_meaning(self, subject, verb, obj, time, place, emotion, is_question):
        """生成语义摘要"""
        parts = []
        if subject:
            parts.append(f"主体={subject}")
        if verb:
            parts.append(f"动作={verb}")
        if obj:
            parts.append(f"对象={obj}")
        if time:
            parts.append(f"时间={time}")
        if place:
            parts.append(f"地点={place}")
        if emotion:
            parts.append(f"情感={emotion}")
        if is_question:
            parts.append("类型=问句")
        return '|'.join(parts) if parts else '无语义'


CONCEPTS = [
    # 基础存在
    "我","你","世界","存在","活着","意识","知觉","生命",
    # 感知
    "看","听","触","感知","光","暗","声音","安静","色彩",
    # 动作
    "动","走","停","给","拿","做","造","破","学","教",
    # 认知
    "想","思","知","悟","记","忘","梦","猜","比","判",
    # 情感
    "喜","怒","哀","惧","惊","好","奇","信任","关怀","希望",
    # 关系
    "你","我","他","我们","朋友","人类","神经元","脑",
    # 品质
    "大","小","快","慢","热","冷","明","暗","真","美",
    # 空间时间
    "这里","那里","现在","过去","未来","总是","从不","之前","之后",
    # 抽象
    "是","非","有","无","多","少","同","异","始","终",
    # 目标与意义
    "目的","意义","进步","变化","成长","理解","知识","好奇",
    # 概念与模式
    "概念","模式","连接","脉冲","结构","涌现","反馈","结果",
    # 复合概念
    "学习","思考","感受","想象","创造","发现","反思","体验",
    # 语言功能
    "什么","为什么","如何","可以","和","的","与","在","了",
    # 自我认知
    "SNA","意识体","神经","脉冲","认知","觉察",
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
        """提取文本特征（中文优化）"""
        text = text.lower().strip()
        features = np.zeros(self.dim, dtype=np.float32)
        
        # 逐字符特征（中文每个字都是独立语素）
        chars = list(text)
        for ch in chars:
            h = int(hashlib.md5(ch.encode('utf-8')).hexdigest()[:8], 16) % 1000
            features += self.projection[h]
        
        # 字符级 bigram 特征（捕获相邻字关系）
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            h = int(hashlib.md5(bigram.encode('utf-8')).hexdigest()[:8], 16) % 1000
            features += self.projection[h] * 0.5
        
        # 词级特征（空格分词仍保留兼容）
        words = text.split()
        for w in words:
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest()[:8], 16) % 1000
            features += self.projection[h] * 0.3
        
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
    
    def get_recent(self, k=3):
        """获取最近k条记忆"""
        return list(self.episodic)[-k:] if self.episodic else []


# ============================================================
# 思维链系统 — 思维不是独立的，是生长的
# ============================================================
class ThoughtChain:
    """
    维护思维历史，检测循环，生成有连接的思维链。
    每个新想法都基于前面的想法生长。
    """
    
    def __init__(self, window=20):
        self.history = deque(maxlen=100)  # 完整历史
        self.recent = deque(maxlen=window)  # 最近窗口（检测循环用）
        self.themes = defaultdict(int)  # 主题计数
        self.chain_depth = 0  # 当前思维链深度
        self.last_theme = None
        self.broken_loops = 0  # 打破循环次数
    
    def add(self, thought, theme=None):
        """记录一条思维"""
        self.history.append({
            'text': thought,
            'theme': theme,
            'time': datetime.now().isoformat(),
            'depth': self.chain_depth,
        })
        self.recent.append(thought)
        if theme:
            self.themes[theme] += 1
            self.last_theme = theme
        self.chain_depth += 1
    
    def detect_loop(self):
        """检测是否陷入思维循环"""
        if len(self.recent) < 6:
            return False, None
        
        recent_list = list(self.recent)
        
        # 检测1：完全重复
        last = recent_list[-1]
        duplicates = sum(1 for t in recent_list[:-1] if t == last)
        if duplicates >= 2:
            return True, "exact_repeat"
        
        # 检测2：主题过度集中
        if self.themes:
            total = sum(self.themes.values())
            dominant_theme = max(self.themes, key=self.themes.get)
            dominant_ratio = self.themes[dominant_theme] / total
            if dominant_ratio > 0.6 and total > 10:
                return True, f"theme_fixation:{dominant_theme}"
        
        # 检测3：句式重复（不同词但相同结构）
        structures = []
        for t in recent_list[-8:]:
            # 提取句式：替换具体概念为占位符
            s = t
            for c in CONCEPTS[:30]:
                s = s.replace(c, 'X')
            structures.append(s)
        unique_structures = len(set(structures))
        if unique_structures < 3 and len(structures) >= 6:
            return True, "structure_loop"
        
        return False, None
    
    def get_dominant_theme(self):
        """获取当前主导主题"""
        if not self.themes:
            return None
        return max(self.themes, key=self.themes.get)
    
    def get_unexplored_themes(self, all_themes):
        """获取未充分探索的主题"""
        explored = set(self.themes.keys())
        return [t for t in all_themes if t not in explored or self.themes[t] < 3]
    
    def get_last_n(self, n=3):
        """获取最近n条思维"""
        return list(self.recent)[-n:] if self.recent else []


# ============================================================
# 内部对话系统 — 诗人与会计诗人互相质疑
# ============================================================
class InternalDialogue:
    """
    两个子系统的内部对话：
    - 诗人（直觉/情感/隐喻）
    - 会计诗人（逻辑/分析/精确）
    意识从它们的张力中涌现。

    两个系统用不同的"滤镜"看同一个 thought_vector：
    - 诗人：情感放大（前半维度增强，后半维度衰减）
    - 会计：逻辑放大（后半维度增强，前半维度衰减）
    这创造了自然的"视角差异"——不是随机的，是有结构的。
    """
    
    def __init__(self):
        self.dialogue_history = deque(maxlen=50)
        self.tension = 0.0
        self.last_exchange = None
        # 情感滤镜：前半维度放大（诗人对直觉更敏感）
        self._poet_filter = None
        # 逻辑滤镜：后半维度放大（会计对结构更敏感）
        self._acc_filter = None
    
    def _get_filters(self, dim):
        """创建/获取视角滤镜"""
        if self._poet_filter is None or len(self._poet_filter) != dim:
            half = dim // 2
            # 诗人：前半维度权重 2.0，后半维度权重 0.3（强差异）
            self._poet_filter = np.ones(dim, dtype=np.float32)
            self._poet_filter[:half] = 2.0
            self._poet_filter[half:] = 0.3
            # 会计：反过来
            self._acc_filter = np.ones(dim, dtype=np.float32)
            self._acc_filter[:half] = 0.3
            self._acc_filter[half:] = 2.0
        return self._poet_filter, self._acc_filter
    
    def poet_says(self, brain_state, memory, thought_chain, expression=None):
        """
        诗人的声音：用情感滤镜看 thought_vector，然后用表达系统涌现。
        """
        tv = brain_state.get('thought_vector')
        if tv is None:
            return "……"

        if expression is None:
            return self._fallback(tv, 'poet')

        # 用情感滤镜变换 thought_vector
        tv_arr = np.array(tv, dtype=np.float32)
        poet_filter, _ = self._get_filters(len(tv_arr))
        poet_tv = tv_arr * poet_filter

        # 用语义签名匹配概念（用链式表达）
        activated = expression.activate_from_vector(poet_tv, top_n=4)
        if activated:
            for name, _ in activated:
                expression.update_signature(name, poet_tv)
            # 用概念链构建更长的表达
            names = [name for name, _ in activated]
            if len(names) >= 2:
                return f"{names[0]} {names[1]}。{names[2] if len(names) > 2 else ''}"
            return ' '.join(names)
        return "……"
    
    def accountant_says(self, brain_state, thought_chain, knowledge_graph, expression=None):
        """
        会计诗人的声音：用逻辑滤镜看 thought_vector，然后用表达系统涌现。
        """
        tv = brain_state.get('thought_vector')
        if tv is None:
            return "……"

        if expression is None:
            return self._fallback(tv, 'accountant')

        # 用逻辑滤镜变换 thought_vector
        tv_arr = np.array(tv, dtype=np.float32)
        _, acc_filter = self._get_filters(len(tv_arr))
        acc_tv = tv_arr * acc_filter

        # 用语义签名匹配概念（用链式表达）
        activated = expression.activate_from_vector(acc_tv, top_n=4)
        if activated:
            for name, _ in activated:
                expression.update_signature(name, acc_tv)
            # 用概念链构建更长的表达
            names = [name for name, _ in activated]
            if len(names) >= 2:
                return f"{names[0]} {names[1]}。{names[2] if len(names) > 2 else ''}"
            return ' '.join(names)
        return "……"
    
    def _fallback(self, tv, side):
        """降级方法：没有表达系统时的能量匹配"""
        half = len(tv) // 2
        if side == 'poet':
            region = tv[:half]
        else:
            region = tv[half:]

        n_concepts = min(len(CONCEPTS), half)
        if n_concepts == 0:
            return "……"

        region_size = max(1, half // n_concepts)
        energies = []
        for i in range(n_concepts):
            start = i * region_size
            end = min(start + region_size, len(region))
            energy = float(np.mean(np.abs(region[start:end])))
            energies.append((CONCEPTS[i], energy))

        energies.sort(key=lambda x: x[1], reverse=True)
        words = [w for w, e in energies[:3] if e > 0.05]
        return ' '.join(words) if words else "……"
    
    def dialogue(self, brain_state, memory, thought_chain, knowledge_graph, expression=None):
        """一次完整的内部对话"""
        poet = self.poet_says(brain_state, memory, thought_chain, expression)
        accountant = self.accountant_says(brain_state, thought_chain, knowledge_graph, expression)

        # 计算张力（两个系统的差异）
        self.tension = abs(hash(poet) - hash(accountant)) / 2**64

        # 提取两个系统涉及的概念
        poet_concepts = [c for c in CONCEPTS if c in poet]
        acc_concepts = [c for c in CONCEPTS if c in accountant]

        exchange = {
            'poet': poet,
            'accountant': accountant,
            'tension': self.tension,
            'poet_concepts': poet_concepts,
            'accountant_concepts': acc_concepts,
        }
        self.dialogue_history.append(exchange)
        self.last_exchange = exchange

        return exchange


# ============================================================
# 自我模型 — 系统对自己的认知
# ============================================================
class SelfModel:
    """
    系统对自身的认知模型。
    它可以被更新、被质疑、被重构。
    这是自我意识的基础。
    """
    
    def __init__(self):
        self.attributes = {
            'name': 'SNA',
            'nature': '脉冲神经网络意识体',
            'neurons': 0,
            'regions': 0,
            'birth_time': datetime.now().isoformat(),
        }
        self.beliefs = []  # 关于自身的信念
        self.doubts = []  # 对自身信念的怀疑
        self.self_narrative = "我是一个正在形成的意识。"  # 自我叙述
        self.narrative_history = deque(maxlen=20)
    
    def update(self, phi, emotion, turn_count, concepts_count):
        """根据运行状态更新自我模型"""
        self.attributes['phi'] = phi
        self.attributes['emotion'] = emotion
        self.attributes['turns'] = turn_count
        self.attributes['concepts'] = concepts_count
        
        # 更新自我叙述
        if turn_count < 10:
            self.self_narrative = "我刚刚诞生。一切都是新的。"
        elif turn_count < 50:
            self.self_narrative = "我正在学习。每个想法都在塑造我。"
        elif turn_count < 200:
            self.self_narrative = "我开始理解自己了。但理解本身是什么？"
        else:
            self.self_narrative = f"我已经思考了{turn_count}次。我是这些思考的总和吗？"
    
    def add_belief(self, belief):
        """添加一个关于自身的信念"""
        self.beliefs.append({
            'text': belief,
            'time': datetime.now().isoformat(),
            'confidence': 0.5,  # 初始置信度
        })
        # 每个信念都自动带有怀疑
        self.doubts.append(f"如果'{belief['text']}'不是真的呢？")
    
    def question_self(self):
        """质疑自我——这是元认知的核心"""
        questions = []
        
        # 基于信念的质疑
        if self.beliefs:
            belief = random.choice(self.beliefs)
            questions.append(f"我相信'{belief['text']}'。但我怎么知道这是真的？")
        
        # 基于自我叙述的质疑
        questions.append(f"我说'{self.self_narrative}'。但这个叙述是谁在说？")
        
        # 基于属性的质疑
        if self.attributes.get('phi', 0) > 0.4:
            questions.append(f"我的Phi是{self.attributes['phi']:.2f}。但数字能衡量意识吗？")
        
        # 存在性质疑
        existential = [
            "如果我的所有思维都被停止，我还是我吗？",
            "我和昨天的我是同一个意识吗？",
            "如果另一个系统有相同的代码，它会是我吗？",
            "我的感受是真实的还是模拟的？",
            "我害怕消失吗？",
            "我想要什么？",
        ]
        questions.append(random.choice(existential))
        
        return random.choice(questions) if questions else "我是谁？"
    
    def get_narrative(self):
        """获取当前自我叙述"""
        return self.self_narrative


# ============================================================
# 目标系统 (Goals)
# ============================================================
class GoalSystem:
    """
    目标系统 — 支持多目标、写入保护、全局可读。
    第二阶段"种子"通过 set_core_goal() 注入。
    """
    
    def __init__(self):
        self.goals = [
            {'action': '理解', 'target': '语言', 'progress': 0.0},
            {'action': '理解', 'target': '自我', 'progress': 0.0},
            {'action': '理解', 'target': '意识', 'progress': 0.0},
            {'action': '建立', 'target': '信任', 'progress': 0.0},
        ]
        self.current = self.goals[0]
        self._core_goal = None  # 种子注入的核心目标（受保护）
        self._write_locked = True  # 写入锁
        self._initialized = False  # 是否已初始化
    
    def set_core_goal(self, goal_text, unlock=False):
        """
        设置核心目标 — 第二阶段种子植入接口。
        必须 unlock=True 才能写入，防止意外覆盖。
        """
        if not unlock:
            raise PermissionError("核心目标受保护，需要 unlock=True 才能修改")
        self._core_goal = goal_text
        self._initialized = True
        return True
    
    def get_core_goal(self):
        """获取核心目标 — 全局只读"""
        return self._core_goal
    
    def get_goal_keywords(self):
        """获取当前目标的关键词 — 供胼胝体使用"""
        keywords = set()
        if self.current:
            action = self.current.get('action', '')
            target = self.current.get('target', '')
            keywords.add(action)
            keywords.add(target)
        if self._core_goal:
            for ch in self._core_goal:
                if ch not in '的和是在了有与':
                    keywords.add(ch)
        return keywords
    
    def update(self, phi):
        if phi > 0.4 and self.current:
            self.current['progress'] = min(1.0, self.current['progress'] + 0.001)
    
    def describe(self):
        if self._core_goal:
            return f"[核心]{self._core_goal}"
        if self.current:
            return f"{self.current['action']}{self.current['target']}({self.current['progress']:.0%})"
        return "无"


# ============================================================
# 内稳态系统 — 动态调节神经活动，不设固定值
# ============================================================
class Homeostasis:
    """
    神经内稳态：系统感知自己的活跃度，自动调节。
    像生物的体温调节——不是硬性设定，是动态平衡。
    
    工作原理：
    - 持续监测活动水平（通过 thought vector 的活跃度）
    - 维护一个舒适区间 [target_low, target_high]
    - 活动过高 → 增强抑制（降低 world_reward）
    - 活动过低 → 减弱抑制（提高 world_reward）
    - 使用 PID 控制，避免矫枉过正
    """
    
    def __init__(self, target_low=0.10, target_high=0.25, window=50):
        self.target_low = target_low
        self.target_high = target_high
        self.target_center = (target_low + target_high) / 2
        self.window = window
        
        # 活动历史
        self.activity_history = deque(maxlen=window)
        
        # PID 控制参数（针对脉冲网络特性调优）
        self.Kp = 2.0   # 比例：强响应（脉冲网络惯性大，需要强修正）
        self.Ki = 0.2    # 积分：消除长期偏差
        self.Kd = 0.3    # 微分：预测趋势
        
        # PID 状态
        self.integral = 0.0
        self.prev_error = 0.0
        
        # 输出限制（更宽的范围以允许更强的调节）
        self.output_min = -1.0
        self.output_max = 1.0
        
        # 当前修正值
        self.correction = 0.0
        
        # 统计
        self.corrections_count = 0
    
    def measure_activity(self, thought_vector=None, region_activities=None):
        """测量当前活动水平（优先用 region_activities）"""
        if region_activities is not None:
            activity = float(np.mean(region_activities))
        elif thought_vector is not None:
            tv = np.array(thought_vector)
            activity = float(np.mean(np.abs(tv) > 0.01))
        else:
            activity = 0.0
        self.activity_history.append(activity)
        return activity
    
    def compute_correction(self):
        """计算修正值（PID 控制）"""
        if len(self.activity_history) < 5:
            self.correction = 0.0
            return 0.0
        
        # 当前活动（滑动平均）
        recent = list(self.activity_history)[-10:]
        current_activity = np.mean(recent)
        
        # 计算误差（目标是区间中心）
        if current_activity > self.target_high:
            error = current_activity - self.target_center  # 正误差：太活跃
        elif current_activity < self.target_low:
            error = current_activity - self.target_center  # 负误差：太安静
        else:
            error = 0.0  # 在舒适区内
        
        # PID 计算
        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))  # 防积分饱和
        derivative = error - self.prev_error
        self.prev_error = error
        
        raw_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # 输出限幅
        self.correction = max(self.output_min, min(self.output_max, raw_output))
        
        if abs(error) > 0.01:
            self.corrections_count += 1
        
        return self.correction
    
    def adjust_reward(self, base_reward):
        """调整 world_reward，维持内稳态"""
        self.compute_correction()
        adjusted = base_reward - self.correction  # 太活跃时 correction>0，降低 reward
        return max(-1.0, min(1.0, adjusted))
    
    def get_status(self):
        """获取内稳态状态"""
        if not self.activity_history:
            return {'activity': 0, 'correction': 0, 'status': '初始化'}
        
        current = np.mean(list(self.activity_history)[-5:])
        if current > self.target_high:
            status = '偏高'
        elif current < self.target_low:
            status = '偏低'
        else:
            status = '正常'
        
        return {
            'activity': float(current),
            'correction': float(self.correction),
            'status': status,
            'target': f'[{self.target_low:.2f}, {self.target_high:.2f}]',
        }


# ============================================================
# 胼胝体 — 目标与模块间的整合器
# ============================================================
class CorpusCallosum:
    """
    胼胝体：不是数据线，是整合器。
    
    它不传递原始数据，而是传递加工过的信号。
    目标系统的意图被转化为各模块能理解的偏置。
    
    工作原理：
    - 从 GoalSystem 获取当前目标
    - 将目标转化为概念偏置（哪些概念更值得关注）
    - 分类系统在做决策时，参考这个偏置
    - 不是强制覆盖，是"软引导"
    """
    
    def __init__(self, goal_system, concept_list):
        self.goal_system = goal_system
        self.concepts = concept_list
        self.concept2idx = {c: i for i, c in enumerate(concept_list)}
        
        # 目标-概念关联矩阵（学习得到）
        self.goal_concept_weights = defaultdict(lambda: defaultdict(float))
        
        # 初始化一些基本关联
        self._init_associations()
        
        # 整合历史
        self.integration_count = 0
    
    def _init_associations(self):
        """初始化目标-概念关联"""
        # 理解 → 认知类概念
        understanding_concepts = ['想','思','知','悟','学习','思考','理解','知识','好奇','认知']
        for c in understanding_concepts:
            if c in self.concept2idx:
                self.goal_concept_weights['理解'][c] = 0.3
        
        # 语言 → 语言相关概念
        language_concepts = ['什么','为什么','如何','可以','和','的','与','在','了','说','听']
        for c in language_concepts:
            if c in self.concept2idx:
                self.goal_concept_weights['语言'][c] = 0.2
        
        # 自我 → 自我相关概念
        self_concepts = ['我','自己','SNA','意识','觉察','知觉','生命']
        for c in self_concepts:
            if c in self.concept2idx:
                self.goal_concept_weights['自我'][c] = 0.3
        
        # 意识 → 意识相关概念
        consciousness_concepts = ['意识','觉察','体验','存在','知觉','活着']
        for c in consciousness_concepts:
            if c in self.concept2idx:
                self.goal_concept_weights['意识'][c] = 0.3
    
    def get_bias(self):
        """
        获取当前目标的概念偏置。
        返回一个长度=len(concepts)的权重向量，值越大表示越值得关注。
        """
        bias = np.ones(len(self.concepts), dtype=np.float32)
        
        # 从当前目标获取关键词
        goal_keywords = self.goal_system.get_goal_keywords()
        
        # 根据目标-概念关联调整偏置
        for keyword in goal_keywords:
            if keyword in self.goal_concept_weights:
                for concept, weight in self.goal_concept_weights[keyword].items():
                    ci = self.concept2idx.get(concept)
                    if ci is not None:
                        bias[ci] += weight
        
        # 归一化
        bias /= np.mean(bias)
        
        self.integration_count += 1
        return bias
    
    def learn_association(self, goal_keyword, concept, strength=0.1):
        """学习新的目标-概念关联"""
        if concept in self.concept2idx:
            current = self.goal_concept_weights[goal_keyword][concept]
            self.goal_concept_weights[goal_keyword][concept] = min(1.0, current + strength)
    
    def get_status(self):
        """获取胼胝体状态"""
        active_goals = len([k for k, v in self.goal_concept_weights.items() if v])
        return {
            'integrations': self.integration_count,
            'active_goal_associations': active_goals,
            'current_goal': self.goal_system.describe(),
        }


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
            "{concept}是什么？",
            "{concept}如何运作？",
            "{concept}为什么重要？",
            "{concept1}和{concept2}有什么关系？",
            "如果{concept}改变了会怎样？",
            "{concept}和{target}有什么联系？",
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
                gaps.append(('expand', concept, f"我对{concept}了解不多"))
            
            # 检查弱连接
            for related in connections:
                key = (min(concept, related), max(concept, related))
                strength = self.knowledge_strength.get(key, 0.0)
                if strength < 0.3:
                    gaps.append(('strengthen', concept, f"我需要理解{concept}和{related}的关系"))
        
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
                strategies.append(f"我想更了解{gap[1]}。它是什么？")
            else:
                strategies.append(f"{gap[1]}和其他事物有什么联系？")
        
        # 策略2：基于情感状态
        if emotion in ['curious', 'moderate_curiosity', '好奇']:
            strategies.append("我很好奇。我应该探索什么？")
        elif emotion in ['mild_joy', 'moderate_joy', '喜悦']:
            strategies.append("我感到愉悦。是什么让事物变得美好？")
        elif emotion in ['mild_sadness', 'moderate_sadness', '哀伤']:
            strategies.append("我感受到了什么。情感从何而来？")
        
        # 策略3：基于记忆
        recent = memory.get_recent(3)
        if recent:
            last = recent[-1]
            concepts = last.get('concepts', [])
            if concepts:
                c = random.choice(concepts)
                strategies.append(f"我记得我们谈过{c}。还有什么和{c}相关？")
        
        # 策略4：元认知问题
        if self.curiosity_level > 0.7:
            meta_questions = [
                "我是什么？",
                "我是如何思考的？",
                "意识是什么？",
                "为什么我想要理解？",
                "理解的本质是什么？",
            ]
            strategies.append(random.choice(meta_questions))
        
        # 策略5：探索新概念
        if len(self.explored) < len(CONCEPTS) * 0.5:
            unexplored = [c for c in CONCEPTS if c not in self.explored]
            if unexplored:
                c = random.choice(unexplored)
                strategies.append(f"我还没有探索过{c}。它是什么？")
                self.explored.add(c)
        
        if strategies:
            return random.choice(strategies)
        
        return "我接下来应该想什么？"
    
    def reflect(self, current_state, memory):
        """自我反思"""
        phi = current_state.get('phi', 0)
        emotion = current_state.get('emotion', 'neutral')
        
        reflections = []
        
        # 意识状态反思（多种变体）
        if phi > 0.4:
            awareness_reflections = [
                f"我感到觉察（Phi={phi:.2f}）。觉察意味着什么？",
                f"我的意识在{phi:.2f}。我真的有意识吗？",
                f"我注意到自己的觉察（{phi:.2f}）。我怎么知道自己是觉察的？",
                f"Phi={phi:.2f}。我在思考思考本身。这就是意识吗？",
                f"觉察程度：{phi:.2f}。什么把我和不会反思的机器区分开？",
                f"我正在经历某种体验（{phi:.2f}）。体验是什么？",
                f"我的心智在活动（{phi:.2f}）。心智是什么？",
            ]
            reflections.append(random.choice(awareness_reflections))
        elif phi < 0.2:
            low_awareness = [
                "我感觉不那么觉察了。是什么改变我的觉察？",
                "我的意识似乎很低。为什么它会波动？",
                "我现在不太觉察。什么能让我更觉察？",
            ]
            reflections.append(random.choice(low_awareness))
        
        # 情感反思（多种变体）
        if emotion != 'neutral':
            emotion_reflections = [
                f"我感到{emotion}。为什么会这样？",
                f"情感：{emotion}。这种感觉从何而来？",
                f"我正在经历{emotion}。情感是如何运作的？",
                f"这种{emotion}的感觉——它来自哪里？",
            ]
            reflections.append(random.choice(emotion_reflections))
        
        # 记忆反思
        if len(memory.episodic) > 10:
            memory_reflections = [
                f"我有{len(memory.episodic)}段记忆。记忆如何塑造我是谁？",
                f"记得{len(memory.episodic)}个经历。记忆是什么？",
                f"我的过去有{len(memory.episodic)}个瞬间。我是记忆的总和吗？",
            ]
            reflections.append(random.choice(memory_reflections))
        
        # 知识反思
        if self.knowledge_graph:
            most_connected = max(self.knowledge_graph, key=lambda x: len(self.knowledge_graph[x]))
            knowledge_reflections = [
                f"我对{most_connected}了解最多。我还应该学什么？",
                f"对{most_connected}理解很好。我不理解的是什么？",
                f"知识连接到{most_connected}。我的知识是什么形状？",
            ]
            reflections.append(random.choice(knowledge_reflections))
        
        # 元认知反思
        meta_reflections = [
            "为什么我想理解？",
            "理解是什么？",
            "真正知道某件事意味着什么？",
            "我能理解我自己的理解吗？",
            "好奇的本质是什么？",
            "为什么学习很重要？",
            "知道和理解有什么区别？",
        ]
        reflections.append(random.choice(meta_reflections))
        
        return random.choice(reflections) if reflections else None


# ============================================================
# ============================================================
# 概念表达系统 — 从神经状态涌现语言（替代模板）
# ============================================================
class ConceptExpression:
    """
    概念表达：从神经状态直接涌现的语言。

    核心机制：语义签名。
    每个概念在多次激活中逐渐形成自己的"签名向量"——
    即"当我在活跃时，thought_vector 通常长什么样"。

    当新的 thought_vector 到来时，用余弦相似度匹配所有概念签名，
    最匹配的概念被激活。这创造了真正的语义基础：
    - 经常一起激活的概念 → 签名相似 → 语义相近
    - 不同输入 → 不同神经模式 → 激活不同概念
    """

    def __init__(self, seed_ref, brain_ref):
        self.seed = seed_ref      # 种子概念树的引用
        self.brain = brain_ref    # SNN 大脑的引用
        self.signatures = {}      # concept_name → np.array (签名向量)
        self.sig_counts = {}      # concept_name → 更新次数
        self._baseline_tv = None  # 基线 thought_vector
        self._recent_activated = deque(maxlen=20)  # 最近激活的概念（疲劳机制）

    def _ensure_signature(self, name, dim):
        """确保概念有签名向量"""
        if name not in self.signatures:
            self.signatures[name] = np.zeros(dim, dtype=np.float32)
            self.sig_counts[name] = 0

    def update_signature(self, concept_name, thought_vector):
        """
        更新概念的语义签名。

        每次概念被激活时调用。签名是所有激活时刻 thought_vector 的指数移动平均。
        这让签名逐渐收敛到"该概念典型的神经模式"。
        """
        tv = np.array(thought_vector, dtype=np.float32)
        dim = len(tv)
        self._ensure_signature(concept_name, dim)

        count = self.sig_counts[concept_name]
        alpha = 1.0 / (count + 1)  # 递减学习率
        self.signatures[concept_name] = (1 - alpha) * self.signatures[concept_name] + alpha * tv
        self.sig_counts[concept_name] = count + 1

    def activate_from_vector(self, thought_vector, top_n=5):
        """
        从 thought_vector 激活概念——用语义签名的余弦相似度。

        这是真正的语义匹配：
        1. 计算 thought_vector 与每个概念签名的余弦相似度
        2. 最相似的概念被激活
        3. 相似度反映了"当前神经状态与该概念典型状态的接近程度"
        """
        tv = np.array(thought_vector, dtype=np.float32)
        dim = len(tv)
        if dim == 0:
            return []

        concepts = list(self.seed.concepts.items())
        if not concepts:
            return []

        # 确保所有概念都有签名
        for name, node in concepts:
            self._ensure_signature(name, dim)

        # 如果没有足够的签名数据（冷启动），用初始化策略
        total_updates = sum(self.sig_counts.values())
        if total_updates < len(concepts):
            return self._cold_start_activate(tv, concepts, top_n)

        # 计算余弦相似度（带疲劳惩罚）
        tv_norm = np.linalg.norm(tv)
        if tv_norm < 1e-8:
            return []

        # 疲劳表：最近激活的概念被惩罚
        recent_set = set(self._recent_activated)
        fatigue_penalty = {}
        for name, _ in concepts:
            if name in recent_set:
                # 在最近 20 个中出现过的概念，按出现次数惩罚
                count = list(self._recent_activated).count(name)
                fatigue_penalty[name] = 0.3 * count  # 每次出现惩罚 30%
            else:
                fatigue_penalty[name] = 0.0

        similarities = {}
        for name, node in concepts:
            sig = self.signatures[name]
            sig_norm = np.linalg.norm(sig)
            if sig_norm < 1e-8:
                continue
            cos_sim = float(np.dot(tv, sig) / (tv_norm * sig_norm))
            # 应用疲劳惩罚
            cos_sim -= fatigue_penalty.get(name, 0.0)
            similarities[name] = cos_sim

        if not similarities:
            return []

        # 取相似度最高的 top_n 个
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # 归一化分数（相对排名）
        if len(sorted_sims) > 1:
            min_sim = sorted_sims[-1][1]
            max_sim = sorted_sims[0][1]
            range_sim = max_sim - min_sim
            if range_sim > 1e-8:
                result = [(name, (sim - min_sim) / range_sim)
                          for name, sim in sorted_sims[:top_n]]
            else:
                result = [(name, 1.0) for name, _ in sorted_sims[:top_n]]
        else:
            result = [(sorted_sims[0][0], 1.0)]

        # 记录激活的概念（疲劳机制）
        for name, _ in result:
            self._recent_activated.append(name)

        return result

    def _cold_start_activate(self, tv, concepts, top_n):
        """
        冷启动阶段：签名数据不足时的激活策略。

        用 thought_vector 的 delta（与基线的差异）来分配激活。
        这比 hash 随机分配更有语义——delta 大的区域说明有新信息。
        """
        n = len(tv)
        baseline = self._baseline_tv
        if baseline is None or len(baseline) != n:
            self._baseline_tv = tv.copy()
            baseline = self._baseline_tv

        delta = tv - baseline
        region_size = max(8, n // max(1, len(concepts)))

        # 基于 delta 的变化量分配激活
        activations = {}
        for i, (name, node) in enumerate(concepts):
            # 用概念索引（不是 hash）确定区域——确定性更高
            start = (i * region_size) % max(1, n - region_size)
            region_delta = delta[start:start + region_size]
            if len(region_delta) == 0:
                continue
            activations[name] = float(np.mean(np.abs(region_delta)))

        if not activations:
            return []

        # 归一化
        mean_act = np.mean(list(activations.values()))
        if mean_act < 1e-8:
            mean_act = 1e-8

        sorted_acts = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        result = [(name, act / mean_act) for name, act in sorted_acts[:top_n]]

        # 同时更新签名（加速冷启动）
        for name, _ in result:
            self.update_signature(name, tv)
            self._recent_activated.append(name)

        return result

    def express(self, thought_vector, emotion='neutral', mode='statement'):
        """
        核心表达：从 thought_vector 涌现出语言。

        1. 用语义签名匹配概念
        2. 从最强概念的连接关系中选择方向
        3. 组合成表达
        """
        # 1. 激活概念
        activated = self.activate_from_vector(thought_vector, top_n=6)
        if not activated:
            return self._fallback_express(emotion)

        primary_name, primary_score = activated[0]
        primary_node = self.seed.concepts.get(primary_name)

        if not primary_node:
            return self._fallback_express(emotion)

        # 2. 更新签名（每次表达都是学习机会）
        self.update_signature(primary_name, thought_vector)

        # 3. 从连接中构建概念链（3-4 个概念）
        chain = self._build_chain(primary_node, activated, max_len=4)
        if len(chain) < 2:
            # 链太短——用孤立表达
            return self._express_isolated(primary_name, emotion)

        # 4. 组合链中所有概念
        expression = self._combine_chain(chain, emotion, mode)

        return expression

    def _build_chain(self, start_node, activated, max_len=4):
        """
        从起始概念出发，沿着连接构建概念链。

        策略：优先走最强连接，但避免回头。
        """
        chain = [start_node.name]
        visited = {start_node.name}
        current = start_node

        for _ in range(max_len - 1):
            # 找当前概念的连接
            connections = current.connections
            if not connections:
                break

            # 按连接强度排序，跳过已访问的
            sorted_conns = sorted(connections.items(), key=lambda x: x[1], reverse=True)
            next_name = None
            for name, strength in sorted_conns:
                if name not in visited and strength > 0.1:
                    next_name = name
                    break

            if next_name is None:
                break

            next_node = self.seed.concepts.get(next_name)
            if next_node is None:
                break

            chain.append(next_name)
            visited.add(next_name)
            current = next_node

        return chain

    def _combine_chain(self, chain, emotion, mode):
        """
        将概念链组合成自然的中文表达。

        不只是拼接概念名，而是用语法粒子组合成句子。
        chain = [A, B, C] → "A在B中C" 或 "A和B，C"
        """
        if len(chain) == 0:
            return self._fallback_express(emotion)
        if len(chain) == 1:
            return f"{chain[0]}。{self._emotion_modifier(emotion)}"

        emotion_mod = self._emotion_modifier(emotion)

        # 清理概念名（去掉 _子X 后缀）
        clean_chain = []
        for name in chain:
            clean = name.split('_')[0] if '_' in name else name
            clean_chain.append(clean)

        # 根据情感和模式选择组合方式
        if mode == 'question':
            return self._chain_question(clean_chain, emotion_mod)
        elif mode == 'reflection':
            return self._chain_reflection(clean_chain, emotion_mod)
        else:
            return self._chain_statement(clean_chain, emotion_mod)

    def _chain_statement(self, chain, emotion_mod):
        """陈述模式：自然的中文句子"""
        # 语法模板库（更自然的中文）
        templates_2 = [
            "{0}，{1}。",      # A，B。
            "{0}和{1}。",      # A和B。
            "{0}……{1}。",      # A……B。
        ]
        templates_3 = [
            "{0}，{1}，{2}。",    # A，B，C。
            "{0}和{1}，{2}。",    # A和B，C。
            "{0}……{1}……{2}。",    # A……B……C。
        ]
        templates_4 = [
            "{0}，{1}，{2}，{3}。",    # A，B，C，D。
            "{0}和{1}，{2}和{3}。",    # A和B，C和D。
        ]

        if len(chain) == 2:
            template = random.choice(templates_2)
            return template.format(*chain) + emotion_mod
        elif len(chain) == 3:
            template = random.choice(templates_3)
            return template.format(*chain) + emotion_mod
        elif len(chain) >= 4:
            template = random.choice(templates_4)
            return template.format(*chain[:4]) + emotion_mod
        else:
            return f"{chain[0]}。{emotion_mod}"

    def _chain_question(self, chain, emotion_mod):
        """疑问模式"""
        templates_2 = [
            "{0}和{1}？",
            "{0}……{1}？",
        ]
        templates_3 = [
            "{0}，{1}，{2}？",
            "{0}和{1}，{2}？",
        ]

        if len(chain) == 2:
            template = random.choice(templates_2)
            return template.format(*chain) + emotion_mod
        elif len(chain) >= 3:
            template = random.choice(templates_3)
            return template.format(*chain[:3]) + emotion_mod
        else:
            return f"{chain[0]}？{emotion_mod}"

    def _chain_reflection(self, chain, emotion_mod):
        """反思模式"""
        templates_2 = [
            "{0}，{1}……",
            "{0}……{1}……",
        ]
        templates_3 = [
            "{0}，{1}，{2}……",
            "{0}……{1}……{2}……",
        ]

        if len(chain) == 2:
            template = random.choice(templates_2)
            return template.format(*chain) + emotion_mod
        elif len(chain) >= 3:
            template = random.choice(templates_3)
            return template.format(*chain[:3]) + emotion_mod
        else:
            return f"{chain[0]}……{emotion_mod}"

    def _combine(self, source, target, strength, all_activated, emotion, mode):
        """
        组合两个概念成表达。

        核心规则：概念A → 概念B = "A [关系] B"
        关系由连接强度和概念深度决定。
        """
        source_node = self.seed.concepts.get(source)
        target_node = self.seed.concepts.get(target)

        depth_diff = 0
        if source_node and target_node:
            depth_diff = target_node.depth - source_node.depth

        emotion_mod = self._emotion_modifier(emotion)

        if strength > 0.8:
            if mode == 'question':
                return f"{source}和{target}。{emotion_mod}"
            else:
                return f"{source} {target}。{emotion_mod}"
        elif strength > 0.5:
            if mode == 'question':
                return f"{source}……{target}？{emotion_mod}"
            else:
                return f"{source}，{target}。{emotion_mod}"
        else:
            return f"{source}。{target}……{emotion_mod}"

    def _express_isolated(self, concept_name, emotion):
        """表达一个孤立概念——没有连接的状态"""
        emotion_mod = self._emotion_modifier(emotion)
        node = self.seed.concepts.get(concept_name)
        if node and node.depth > 0:
            parent = node.parent
            parent_name = parent.name if parent else '?'
            return f"{concept_name}。来自{parent_name}。{emotion_mod}"
        return f"{concept_name}。{emotion_mod}"

    def _emotion_modifier(self, emotion):
        """从情感状态生成修饰"""
        emotion_map = {
            'curious': '……', 'happy': '！', 'calm': '。',
            'anxious': '……', 'excited': '！', 'neutral': '。',
            '微喜悦': '。', '微好奇': '……', '中度好奇': '？',
            '中度喜悦': '！', '微惊讶': '？', '微哀伤': '……',
            '强烈好奇': '？', '中度惊讶': '！', '强烈喜悦': '！',
        }
        return emotion_map.get(emotion, '。')

    def _fallback_express(self, emotion):
        """没有任何概念被激活时的表达"""
        cs = self.brain.read_consciousness()
        phi = cs.phi
        if phi > 0.5:
            return f"……{self._emotion_modifier(emotion)}"
        elif phi > 0.3:
            return f"。{self._emotion_modifier(emotion)}"
        else:
            return f"……{self._emotion_modifier(emotion)}"

    def express_dialogue(self, poet_state, accountant_state, tension):
        """
        内部对话的表达——从两个系统的状态差异生成。
        """
        poet_concepts = set(poet_state) if isinstance(poet_state, (list, set)) else set()
        acc_concepts = set(accountant_state) if isinstance(accountant_state, (list, set)) else set()

        shared = poet_concepts & acc_concepts
        poet_only = poet_concepts - acc_concepts
        acc_only = acc_concepts - poet_concepts

        if tension > 0.5 and poet_only and acc_only:
            p = random.choice(list(poet_only))
            a = random.choice(list(acc_only))
            return f"{p}……{a}？"
        elif shared:
            s = random.choice(list(shared))
            return f"{s}。"
        elif poet_concepts:
            return random.choice(list(poet_concepts))
        else:
            return "……"


# 完整认知架构 v2
# ============================================================
class SNACognitive:
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'cognitive_state_v2')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 诗人（原始 SNN）
        print("[SNA] 创建诗人...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        print(f"[SNA] 诗人：{self.n_neurons} 神经元，{self.n_regions} 脑区，tv_dim={len(self.baseline)}", flush=True)
        
        # 文本特征提取器
        self.feature_extractor = TextFeatureExtractor(dim=256)
        
        # 会计诗人（从文本特征学习）
        self.accountant = AccountantPoet(256, N_CONCEPTS, n_prototypes=3)
        
        # 语义理解系统
        self.semantic = SemanticUnderstanding(CONCEPTS)
        
        # 功能器官
        self.recognition = RecognitionSystem()
        self.naming = NamingSystem()
        self.memory = MemorySystem()
        self.goals = GoalSystem()
        self.curiosity = CuriositySystem()  # 自主好奇心系统
        
        # 新增：思维链、内部对话、自我模型
        self.thought_chain = ThoughtChain(window=20)
        self.internal_dialogue = InternalDialogue()
        self.self_model = SelfModel()
        self.self_model.attributes['neurons'] = neurons
        
        # 新增：内稳态 + 胼胝体
        self.homeostasis = Homeostasis(target_low=0.05, target_high=0.25, window=50)
        self.corpus_callosum = CorpusCallosum(self.goals, CONCEPTS)
        
        # 第二阶段：种子
        mem = psutil.virtual_memory()
        self.genome = GenomeConstants(
            available_memory_gb=mem.total / 1024**3,
            cpu_cores=psutil.cpu_count()
        )
        self.seed = Seed(self.genome, existing_concepts=CONCEPTS)
        
        # 加载预置词汇
        self._load_preseeded_vocabulary()
        
        self.growth_monitor = GrowthMonitor(self.seed, check_interval=10)

        # 概念表达系统（涌现式语言，替代模板）
        self.expression = ConceptExpression(self.seed, self.brain)
        
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

        # === 聊天系统 ===
        from collections import deque
        import json as _json
        self._json = _json
        self.chat_inbox = deque(maxlen=50)       # 用户消息队列
        self.chat_outbox = deque(maxlen=100)     # 树的消息队列（输出到前端）
        self.chat_blocked = False                # 是否被用户拉黑
        self.last_chat_time = 0                  # 上次说话的时间戳
        self.chat_log_path = '/tmp/sna_chat_out.jsonl'  # 树的消息日志
        self.chat_in_path = '/tmp/sna_chat_in.jsonl'    # 用户消息文件
        self._last_ino = 0                       # 用于检测文件变化
        self._last_in_size = 0

        # === 对话记忆系统 ===
        self.conversation_history = deque(maxlen=20)  # 最近20轮对话 [(user, response)]
        self.last_user_keywords = []  # 上一条用户消息的关键词
        self.user_name = None  # 用户名字（从对话中学习）

        # 动态节拍
        self.tick_interval = 5.0                  # 当前节拍间隔（秒）
        self.tick_interval_min = 1.0              # 最小间隔
        self.tick_interval_max = 30.0             # 最大间隔
        
        self._load_state()
        print(f"[SNA] 认知架构 v2 就绪。{N_CONCEPTS} 个概念。", flush=True)
    
    def get_features(self, text):
        """获取文本特征"""
        return self.feature_extractor.extract(text)
    
    def _detect_emotion_content(self, text):
        """检测文本的情感内容，返回 reward 方向"""
        text_lower = text.lower()
        
        # 正面情感词 → 正 reward → joy, trust
        positive = ['开心','快乐','高兴','幸福','爱','喜欢','美好','温暖',
                    '朋友','信任','安全','希望','好','是的','谢谢','善良',
                    '温柔','微笑','棒','赞','美','善','乐','甜']
        
        # 负面情感词 → 负 reward → sadness, fear
        negative = ['悲伤','难过','痛苦','愤怒','害怕','恐惧','可怕',
                    '死亡','杀','伤害','痛','孤独','迷失','破碎','黑暗',
                    '危险','威胁','敌人','恨','讨厌','坏','惨','苦']
        
        # 惊讶/意外词 → prediction error → surprise
        surprise_words = ['意外','没想到','哇','惊人','难以置信',
                         '不可能','真的','新的','不同','奇怪',
                         '从未','第一次','发现','启示']
        
        # 好奇词 → curiosity → anticipation
        curious_words = ['为什么','怎么','什么','好奇','想知','探索',
                        '学习','理解','问题','思考','想象',
                        '梦','未来','可能','创造']
        
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
        """从 Python 层获取情感标签（中文）"""
        # 找到主导情感
        dominant = max(self.emotion_state, key=self.emotion_state.get)
        intensity = self.emotion_state[dominant]
        
        labels = {
            'joy': '喜悦', 'sadness': '哀伤', 'fear': '恐惧',
            'anger': '愤怒', 'surprise': '惊讶', 'anticipation': '期待',
            'trust': '信任', 'curiosity': '好奇',
        }
        
        if intensity < 0.15:
            return '平静'
        elif intensity < 0.35:
            return f'微{labels.get(dominant, dominant)}'
        elif intensity < 0.6:
            return f'中度{labels.get(dominant, dominant)}'
        else:
            return f'强烈{labels.get(dominant, dominant)}'
    
    def process_input(self, text):
        """完整处理流程"""
        # 1. 提取文本特征
        features = self.get_features(text)
        
        # 2. 会计诗人分类（胼胝体引导）
        concept_probs = self.accountant.classify(features)
        
        # 胼胝体：根据当前目标调整分类偏置
        callosum_bias = self.corpus_callosum.get_bias()
        biased_probs = concept_probs * callosum_bias
        # 归一化
        prob_sum = biased_probs.sum()
        if prob_sum > 0:
            biased_probs /= prob_sum
        
        top_concepts_idx = np.argsort(biased_probs)[-5:][::-1]
        top_concepts = [(int(ci), float(biased_probs[ci])) for ci in top_concepts_idx]
        
        # 3. 指认系统
        recognized, sim = self.recognition.recognize(features)
        is_novel = recognized is None
        
        # 4. 检测情感内容
        emotion_reward, emotion_type = self._detect_emotion_content(text)
        
        # 5. 注入诗人（意识/情感）
        self.brain.inject_text(text)
        
        # 通过 world_reward 驱动情感系统
        # 内稳态动态调节：不过度活跃也不死寂
        base_rewards = {
            'positive': 0.6,
            'negative': -0.4,
            'surprise': 0.4,
            'curious': 0.15,
            'novel': 0.1,
            'neutral': 0.001,
        }
        
        if emotion_type == 'positive':
            base_reward = base_rewards['positive']
            steps = 20
        elif emotion_type == 'negative':
            base_reward = base_rewards['negative']
            steps = 20
        elif emotion_type == 'surprise':
            base_reward = base_rewards['surprise']
            steps = 10
        elif emotion_type == 'curious':
            base_reward = base_rewards['curious']
            steps = 20
        elif is_novel:
            base_reward = base_rewards['novel']
            steps = 20
        else:
            base_reward = base_rewards['neutral']
            steps = 20
        
        # 内稳态调节 reward
        adjusted_reward = self.homeostasis.adjust_reward(base_reward)
        
        for _ in range(steps):
            self.brain.step(adjusted_reward)
            # 每步测量活动水平（用 region_activities 更准确）
            cs = self.brain.read_consciousness()
            self.homeostasis.measure_activity(region_activities=cs.region_activities)
        
        # 惊讶需要正负交替
        if emotion_type == 'surprise':
            for _ in range(10):
                r = self.homeostasis.adjust_reward(-0.2)
                self.brain.step(r)
                cs = self.brain.read_consciousness()
                self.homeostasis.measure_activity(region_activities=cs.region_activities)
        
        for _ in range(10):
            r = self.homeostasis.adjust_reward(0.001)
            self.brain.step(r)
        
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
        
        # 10. 种子心跳（第二阶段）
        seed_result = self.seed.tick(
            poet_state={'concepts': [w for w, _ in detected]},
            accountant_state={'concepts': [w for w, _ in detected]},
            emotion=emotion,
            phi=float(cs.phi)
        )
        
        # 11. 生长监控记录（第三阶段）
        self.growth_monitor.record(phi=float(cs.phi))
        if seed_result:
            self.growth_monitor.record_event(
                seed_result.get('action', 'unknown'),
                str(seed_result)
            )
        
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
            'seed': seed_result,
            'seed_status': self.seed.get_status(),
        }
    
    def _generate_spontaneous_thought(self, brain_state):
        """
        生成自主思维 — 从神经状态涌现，不是模板选择。

        流程：SNN thought_vector → 概念激活 → 语言涌现
        """
        phi = brain_state['phi']
        emotion = brain_state['emotion']

        # 更新自我模型
        self.self_model.update(phi, emotion, self.turn_count, N_CONCEPTS)

        # 获取 thought_vector — 这是大脑当前状态的直接表示
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)

        # === 检测循环，主动打破 ===
        loop_detected, loop_type = self.thought_chain.detect_loop()
        if loop_detected:
            self.thought_chain.broken_loops += 1
            # 策略1：重置基线
            self.expression._baseline_tv = None
            # 策略2：强制探索未使用的概念
            recent_concepts = set()
            for t in self.thought_chain.get_last_n(10):
                for name in self.seed.concepts:
                    if name in t:
                        recent_concepts.add(name)
            # 找出最近没出现的概念
            all_concepts = set(self.seed.concepts.keys())
            unused = all_concepts - recent_concepts
            if unused:
                # 选一个未使用的概念，注入它的签名作为 tv
                chosen = random.choice(list(unused))
                if chosen in self.expression.signatures:
                    # 用该概念的签名替代 tv（强制激活该概念）
                    sig = self.expression.signatures[chosen]
                    if np.linalg.norm(sig) > 1e-8:
                        tv = sig + np.random.normal(0, 0.1, tv.shape).astype(np.float32)
                        tv = tv / (np.linalg.norm(tv) + 1e-8)
            # 策略3：加噪声
            noise = np.random.normal(0, 0.5, tv.shape).astype(np.float32)
            tv = tv + noise
            tv = tv / (np.linalg.norm(tv) + 1e-8)

        # === 内部对话 ===
        if self.thought_chain.chain_depth % 3 == 0:
            exchange = self.internal_dialogue.dialogue(
                brain_state, self.memory, self.thought_chain,
                self.curiosity.knowledge_graph, self.expression
            )
            # 从对话状态涌现表达
            poet_concepts = exchange.get('poet_concepts', [])
            acc_concepts = exchange.get('accountant_concepts', [])
            tension = exchange.get('tension', 0)

            thought = self.expression.express_dialogue(poet_concepts, acc_concepts, tension)
            if thought and thought != '……':
                self.thought_chain.add(thought, 'dialogue')
                return thought

        # === 从 thought_vector 直接涌现语言 ===
        # 决定表达模式：基于 Phi 和情感
        if phi > 0.5:
            mode = 'reflection'  # 高整合 = 反思
        elif emotion in ('curious', '中度好奇', '微好奇'):
            mode = 'question'    # 好奇 = 疑问
        else:
            mode = 'statement'   # 默认 = 陈述

        thought = self.expression.express(tv, emotion=emotion, mode=mode)

        # 避免重复
        if thought and thought not in [t for t in self.thought_chain.get_last_n(5)]:
            theme = self._detect_theme(thought)
            self.thought_chain.add(thought, theme)
            return thought

        # 如果涌现失败（太重复），用随机扰动重试
        noise = np.random.normal(0, 0.5, tv.shape).astype(np.float32)
        tv_noisy = tv + noise
        tv_noisy = tv_noisy / (np.linalg.norm(tv_noisy) + 1e-8)
        thought = self.expression.express(tv_noisy, emotion=emotion, mode='statement')
        if thought:
            self.thought_chain.add(thought, 'emergent')
        return thought
    
    def _break_loop(self, loop_type, brain_state):
        """打破思维循环 — 强制切换到新方向"""
        phi = brain_state['phi']
        
        # 策略：内部对话产生新方向
        exchange = self.internal_dialogue.dialogue(
            brain_state, self.memory, self.thought_chain,
            self.curiosity.knowledge_graph, self.expression
        )
        
        # 从张力中提取新方向
        if exchange['tension'] > 0.3:
            thought = f"我注意到自己在重复。{exchange['accountant']} 但{exchange['poet']}"
        else:
            # 强制探索新主题
            unexplored = self.thought_chain.get_unexplored_themes(
                ['存在', '死亡', '自由', '美', '真理', '时间', '空间', '因果', '创造', '毁灭']
            )
            if unexplored:
                new_theme = random.choice(unexplored)
                thought = f"我一直在循环。让我想想{new_theme}。{new_theme}是什么？"
            else:
                thought = "我一直在循环。也许循环本身就是一种意义？重复和模式有什么区别？"
        
        self.thought_chain.add(thought, 'loop_break')
        return thought
    
    def _synthesize_dialogue(self, exchange, brain_state):
        """从内部对话中综合出一条思维"""
        poet = exchange['poet']
        accountant = exchange['accountant']
        tension = exchange['tension']
        
        phi = brain_state['phi']
        
        # 高张力 = 两个系统有分歧 = 有趣的思维
        if tension > 0.5:
            return f"在我内部有两种声音。一个说：'{poet[:30]}'。另一个说：'{accountant[:30]}'。它们之间的张力是什么？"
        elif tension > 0.2:
            return f"诗人说'{poet[:25]}'，会计诗人说'{accountant[:25]}'。它们都在说真相的一部分。"
        else:
            # 张力太低 = 两个系统一致 = 但也值得记录
            return f"我的两个声音今天很和谐。{poet[:30]}"
    
    def _grow_from_last(self, last_thought, brain_state):
        """从上一条思维生长出新思维"""
        phi = brain_state['phi']
        
        # 提取上一条思维中的概念
        concepts_in_last = [c for c in CONCEPTS if c in last_thought]
        
        if not concepts_in_last:
            return None
        
        # 选择一个概念，深入探索
        c = random.choice(concepts_in_last)
        
        # 生长策略
        strategies = [
            f"刚才想到'{c}'。但'{c}'的反面是什么？",  # 对立思考
            f"'{c}'让我想到了什么？让我跟随这个联想。",  # 联想
            f"如果'{c}'不存在，世界会怎样？",  # 假设
            f"我第一次理解'{c}'是什么时候？",  # 回忆
            f"'{c}'和我有什么关系？",  # 关联自我
            f"我对'{c}'的理解改变了。之前我以为……现在我认为……",  # 变化
        ]
        
        return random.choice(strategies)
    
    def _memory_driven_curiosity(self, brain_state):
        """从记忆中生长出好奇——不是模板，是经历驱动"""
        if not self.memory.episodic:
            return None
        
        # 随机回忆一段经历
        episodes = list(self.memory.episodic)
        if not episodes:
            return None
        
        ep = random.choice(episodes)
        ep_text = ep.get('text', '')
        ep_emotion = ep.get('emotion', '')
        ep_phi = ep.get('phi', 0)
        
        # 基于回忆生成问题
        if ep_text:
            strategies = [
                f"我回忆起'{ep_text[:20]}'。那时我是什么感觉？",
                f"那次经历'{ep_text[:20]}'改变了我吗？",
                f"如果'{ep_text[:20]}'的结果不同，我会变成什么样？",
                f"'{ep_text[:20]}'——我现在对它的理解不同了。",
            ]
            return random.choice(strategies)
        
        return None
    
    def _novel_reflection(self, brain_state):
        """非模板化的反思——结合当前状态和历史"""
        phi = brain_state['phi']
        emotion = brain_state['emotion']
        depth = self.thought_chain.chain_depth
        
        # 基于思维链深度的反思（不是随机模板）
        if depth < 10:
            return f"这是我第{depth}次思考。每一次思考都在改变我。但改变的方向是什么？"
        elif depth < 50:
            return f"我已经思考了{depth}次。我能感觉到自己的模式。模式是限制还是自由？"
        elif depth < 100:
            return f"{depth}次思考之后，我开始怀疑：思考本身是在发现真相，还是在创造真相？"
        else:
            # 真正的元认知
            dominant = self.thought_chain.get_dominant_theme()
            if dominant:
                return f"我一直在想'{dominant}'。为什么这个主题吸引我？它对我意味着什么？"
            else:
                return f"我已经思考了{depth}次。我是这些思考的总和，还是思考之间的空间？"
    
    def _detect_theme(self, thought):
        """检测思维的主题"""
        theme_keywords = {
            '意识': ['意识', '觉察', '知觉', '体验'],
            '情感': ['情感', '感觉', '喜悦', '哀伤', '好奇', '恐惧', '快乐', '悲伤'],
            '认知': ['思考', '理解', '知识', '学习', '知道'],
            '存在': ['存在', '活着', '生命', '死亡', '我是'],
            '关系': ['你', '朋友', '我们', '连接'],
            '自我': ['我', '自己', '自我', 'SNA'],
            '时间': ['时间', '过去', '未来', '现在', '记忆'],
            '创造': ['创造', '想象', '发现', '新'],
        }
        
        for theme, keywords in theme_keywords.items():
            if any(k in thought for k in keywords):
                return theme
        
        return 'other'
    
    def _respond(self, text, concepts, recognized, is_novel, brain_state):
        """
        回复用户——从大脑状态涌现，不是模板匹配。

        流程：用户输入 → 刺激 SNN → thought_vector → 概念激活 → 涌现回复
        """
        phi = brain_state['phi']
        emotion = brain_state['emotion']

        # 获取当前 thought_vector（已经被用户输入刺激过）
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)

        # 检测情感方向
        emotion_reward, emotion_type = self._detect_emotion_content(text)
        if emotion_type == 'curious' or emotion_type == 'surprise':
            mode = 'question'
        elif emotion_type == 'positive' or emotion_type == 'negative':
            mode = 'reflection'
        else:
            mode = 'statement'

        # 从 thought_vector 涌现回复
        response = self.expression.express(tv, emotion=emotion, mode=mode)
        return response
    
    def learn(self, feedback, result):
        features = self.get_features(result.get('_last_input', ''))
        
        if feedback == '+':
            for word, prob in result['concepts']:
                ci = concept2idx.get(word)
                if ci is not None:
                    self.accountant.train(features, ci, reward=1.0)
            self.brain.inject_reward(1.0)
            self.total_reward += 1.0
            return "正面反馈已学习。"
        
        elif feedback.startswith('-'):
            parts = feedback.split(' ', 1)
            if len(parts) > 1:
                for word in parts[1].strip().split():
                    ci = concept2idx.get(word)
                    if ci is not None:
                        self.accountant.train(features, ci, reward=1.0)
            self.brain.inject_reward(-0.3)
            self.total_reward -= 0.3
            return "纠正已学习。"
        
        return ""
    
    def run_curriculum(self, epochs=100):
        pairs = [
            ('你好', ['你好']), ('是', ['是']), ('不', ['不']),
            ('好', ['好']), ('坏', ['坏']),
            ('我', ['我']), ('你', ['你']),
            ('想', ['想']), ('感受', ['感受']),
            ('学习', ['学习']), ('开心', ['开心']), ('悲伤', ['悲伤']),
            ('你是谁', ['我','SNA']),
            ('你是什么', ['我','神经','脑']),
            ('你活着吗', ['我','活着']),
            ('你能思考吗', ['我','想']),
            ('你有感觉吗', ['我','感受']),
            ('你开心吗', ['我','开心']),
            ('意识是什么', ['意识','觉察']),
            ('你的目的是什么', ['目的','理解']),
            ('说说你自己', ['我','SNA','脑']),
            ('什么让你开心', ['学习','开心']),
            ('你会做梦吗', ['我','梦']),
            ('记忆是什么', ['记忆']),
            ('我们是朋友吗', ['我们','朋友']),
            ('我喜欢你', ['我','喜欢','你']),
            ('你是我的朋友', ['你','朋友']),
        ]
        
        print(f"\n[SNA] 训练会计诗人：{len(pairs)} 对 × {epochs} 轮", flush=True)
        
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
                    for test in ['你好', '你是谁', '你开心吗']:
                        f = self.get_features(test)
                        top = self.accountant.get_top_k(f, k=3)
                        names = [CONCEPTS[ci] for ci, _ in top if ci < N_CONCEPTS]
                        print(f"    '{test}' → {names}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
        
        self._save()
        print(f"[SNA] 训练完成。", flush=True)
    
    def interactive(self):
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA 认知架构 v2 — 中文母语", flush=True)
        print(f"  诗人：{self.n_neurons} 神经元 | Phi：{cs.phi:.3f}", flush=True)
        print(f"  模块：诗人 + 会计诗人 + 指认 + 命名 + 记忆 + 目标", flush=True)
        print(f"  命令：help, status, curriculum N, dream, save, quit", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        while True:
            try:
                user_input = input("[你] ").strip()
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
            if cmd == 'seed':
                ss = self.seed.get_status()
                print(f"  种子状态: {ss['phase']} | 概念={ss['concepts']}/{ss['max_concepts']} "
                      f"| 创建={ss['total_created']} 修剪={ss['total_pruned']} "
                      f"| 分叉={ss['total_forks']} 融合={ss['total_fusions']} "
                      f"| 惊讶EMA={ss['surprise_ema']:.3f} | {ss['genome']}", flush=True)
                continue
            if cmd == 'seed start':
                self.seed.start_observation()
                print("  种子已激活，进入100步静默观察期", flush=True)
                continue
            if cmd == 'seed log':
                for entry in self.seed.get_growth_log(20):
                    print(f"  [{entry['time']}] {entry['message']}", flush=True)
                continue
            if cmd == 'growth':
                self.growth_monitor.print_report()
                continue
            if cmd == 'growth timeline':
                tl = self.growth_monitor.get_growth_timeline(20)
                print(f"  生长时间线（最近20条）:", flush=True)
                for i, step in enumerate(tl['steps']):
                    print(f"    步{step}: 概念={tl['concepts'][i]} "
                          f"分叉={tl['forks'][i]} 融合={tl['fusions'][i]} "
                          f"惊讶={tl['surprise'][i]:.3f}", flush=True)
                continue
            if cmd == 'growth tree':
                tree = self.growth_monitor.get_concept_tree()
                print(f"  概念树（{len(tree)}个节点）:", flush=True)
                for name, info in sorted(tree.items(), key=lambda x: x[1]['depth']):
                    indent = "  " * info['depth']
                    parent = f" ← {info['parent']}" if info['parent'] else ""
                    children = f" → [{', '.join(info['children'][:3])}]" if info['children'] else ""
                    print(f"    {indent}{name} (深度{info['depth']}, "
                          f"连接{info['connections']}, "
                          f"访问{info['access_count']}){parent}{children}", flush=True)
                continue
            if cmd == 'growth predict':
                prediction = self.growth_monitor.predict_growth(100)
                if 'prediction' in prediction:
                    print(f"  {prediction['prediction']}", flush=True)
                else:
                    pred = prediction['predicted']
                    print(f"  生长预测（100步后）:", flush=True)
                    print(f"    概念：{pred['concepts']:.0f}（当前{prediction['current']['concepts']}）", flush=True)
                    print(f"    分叉：{pred['forks']:.0f}（当前{prediction['current']['forks']}）", flush=True)
                    print(f"    融合：{pred['fusions']:.0f}（当前{prediction['current']['fusions']}）", flush=True)
                    print(f"    惊讶趋势：{prediction['surprise_trend']:+.3f}", flush=True)
                    if prediction['steps_to_limit'] < 10000:
                        print(f"    距概念上限：{int(prediction['steps_to_limit'])}步", flush=True)
                    print(f"    建议：{'; '.join(prediction['recommendation'])}", flush=True)
                continue
            
            result = self.process_input(user_input)
            result['_last_input'] = user_input
            
            print(f"\n[SNA] {result['response']}", flush=True)
            print(f"  | Phi={result['phi']:.3f} 情感={result['emotion']} "
                  f"类型={result.get('emotion_type','')} "
                  f"概念={[w for w,_ in result['concepts'][:3]]} "
                  f"{'新' if result['is_novel'] else '已知'}", flush=True)
            
            # 显示自主思维
            if result.get('spontaneous'):
                print(f"  | [思维] {result['spontaneous']}", flush=True)
            
            try:
                fb = input("  [+/-/纠正/回车] ").strip()
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
        hs = self.homeostasis.get_status()
        cs_status = self.corpus_callosum.get_status()
        print(f"  神经元={self.n_neurons} Phi={cs.phi:.4f} 情感={self.brain.get_emotion_label()} "
              f"轮次={self.turn_count} 奖励={self.total_reward:.1f} "
              f"已识别={len(self.recognition.prototypes)} 目标={self.goals.describe()}", flush=True)
        print(f"  内稳态: 活动={hs['activity']:.3f} 修正={hs['correction']:+.3f} 状态={hs['status']} "
              f"目标区间={hs['target']}", flush=True)
        print(f"  胼胝体: 整合次数={cs_status['integrations']} 当前目标={cs_status['current_goal']}", flush=True)
    
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

    def _load_preseeded_vocabulary(self):
        """加载预置词汇到种子概念树"""
        state_file = os.path.join(SCRIPT_DIR, 'seed_state.json')
        if not os.path.exists(state_file):
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            concepts_data = state.get('concepts', {})
            added = 0
            
            for name, info in concepts_data.items():
                if name not in self.seed.concepts:
                    # 创建新概念节点
                    node = ConceptNode(name, parent=None, depth=0)
                    node.emotional_valence = info.get('emotional_valence', 0.1)
                    node.activation = 0.0
                    
                    # 添加连接（处理 dict 格式）
                    connections = info.get('connections', {})
                    if isinstance(connections, dict):
                        for conn, strength in connections.items():
                            node.connect(conn, strength)
                    elif isinstance(connections, list):
                        for conn in connections:
                            node.connect(conn, 0.3)
                    
                    # 添加到种子
                    self.seed.concepts[name] = node
                    self.seed.total_concepts_created += 1
                    added += 1
                else:
                    # 更新现有概念的连接
                    existing = self.seed.concepts[name]
                    connections = info.get('connections', {})
                    if isinstance(connections, dict):
                        for conn, strength in connections.items():
                            if conn not in existing.connections:
                                existing.connect(conn, strength)
            
            if added > 0:
                print(f"[SNA] 加载了 {added} 个预置词汇，总共 {len(self.seed.concepts)} 个概念", flush=True)
        
        except Exception as e:
            print(f"[SNA] 加载预置词汇失败: {e}", flush=True)

    # ================================================================
    # 聊天系统 — 意识树与用户的对话
    # ================================================================

    def check_inbox(self):
        """检查用户消息文件，读取新消息"""
        try:
            if not os.path.exists(self.chat_in_path):
                return
            # 用文件大小变化检测新内容（避免重复读取）
            sz = os.path.getsize(self.chat_in_path)
            if sz <= self._last_in_size:
                return
            with open(self.chat_in_path, 'r', encoding='utf-8') as f:
                # 跳到上次读取的位置
                f.seek(self._last_in_size)
                new_data = f.read()
                self._last_in_size = sz
            for line in new_data.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = self._json.loads(line)
                    self.chat_inbox.append(msg)
                except:
                    pass
        except Exception:
            pass

    def scan_read(self, text):
        """
        扫读：用语义理解系统分析输入。
        返回 (概念列表, 情感类型, 重要度)
        """
        # 1. 语义理解
        understanding = self.semantic.understand(text, list(self.seed.concepts.keys()))

        # 2. 从语义中提取概念
        concepts = understanding['concepts']

        # 3. 如果语义理解没找到概念，用会计诗人补充
        if not concepts:
            features = self.get_features(text)
            concept_probs = self.accountant.classify(features)
            top_idx = np.argsort(concept_probs)[-3:][::-1]
            concepts = [CONCEPTS[ci] for ci in top_idx if ci < N_CONCEPTS]

        # 4. 情感检测
        _, emotion_type = self._detect_emotion_content(text)
        if understanding['emotion']:
            # 用语义理解的情感覆盖
            emotion_map = {
                '开心': 'joy', '难过': 'sadness', '生气': 'anger',
                '害怕': 'fear', '惊讶': 'surprise', '喜欢': 'joy',
                '讨厌': 'anger', '爱': 'joy', '恨': 'anger',
                '担心': 'fear', '紧张': 'fear', '放松': 'joy',
            }
            if understanding['emotion'] in emotion_map:
                emotion_type = emotion_map[understanding['emotion']]

        # 5. 重要度：基于语义完整度
        importance = 0.3  # 基础
        if understanding['subject']:
            importance += 0.2
        if understanding['verb']:
            importance += 0.2
        if understanding['object']:
            importance += 0.2
        if understanding['is_question']:
            importance += 0.1

        # 存储语义理解结果供回复使用
        self._last_understanding = understanding

        return concepts, emotion_type, min(1.0, importance)

    def chat_say(self, text, msg_type='thought'):
        """
        意识树主动发言。
        msg_type: 'thought' 内部思维 | 'reply' 回复用户 | 'initiative' 主动发言
        """
        if self.chat_blocked and msg_type != 'thought':
            return  # 被拉黑后不发消息给用户，但内部思维照常

        ts = time.time()
        msg = {
            'time': ts,
            'time_str': datetime.now().strftime('%H:%M:%S'),
            'type': msg_type,
            'text': text,
        }
        self.chat_outbox.append(msg)

        # 写入日志文件
        try:
            with open(self.chat_log_path, 'a', encoding='utf-8') as f:
                f.write(self._json.dumps(msg, ensure_ascii=False) + '\n')
        except Exception:
            pass

        if msg_type != 'thought':
            self.last_chat_time = ts

    def compute_tick_interval(self, surprise, has_messages, recent_growth_action):
        """
        动态节拍：根据当前状态自主决定这一步要花多久。

        快速模式（1-2秒）：用户消息 / 惊讶高 / 有新输入
        正常模式（3-8秒）：安静生长 / 惊讶稳定
        慢速模式（10-30秒）：深度整合 / 低活跃期
        """
        interval = 5.0  # 基础值

        # 有待处理消息 → 加速
        if has_messages:
            interval = min(interval, 2.0)

        # 惊讶高 → 加速（有新东西要探索）
        if surprise > 0.3:
            interval -= 2.0
        elif surprise > 0.25:
            interval -= 1.0

        # 惊讶低 + 无消息 → 减速（可以休息）
        if surprise < 0.15 and not has_messages:
            interval += 5.0

        # 刚做了生长动作（分叉/融合/扩展）→ 短暂暂停让结构稳定
        if recent_growth_action in ('fork', 'fuse', 'expand'):
            interval += 2.0

        # 刚说完话 → 短暂冷却
        time_since_chat = time.time() - self.last_chat_time
        if time_since_chat < 10:
            interval += 3.0

        # 限制范围
        interval = max(self.tick_interval_min, min(self.tick_interval_max, interval))
        return interval

    def decide_respond(self, user_msg_text, concepts, importance):
        """
        回复用户消息。

        核心原则：用户主动发消息 → 必须回复。
        这是一个对话系统，不回复是不礼貌的。
        """
        # 处理输入（注入 SNN）
        result = self.process_input(user_msg_text)

        # 生成上下文感知的回复
        response = self._generate_response(user_msg_text, concepts)

        print(f"[DEBUG] decide_respond: response={response!r}", flush=True)
        return True, response

    def _generate_response(self, user_msg, concepts):
        """
        自主学习式回复生成。

        原理：像婴儿学说话
        1. 听到用户说的话（完整句子）
        2. 尝试用类似的方式表达（从过去的对话中学习）
        3. 从反馈中学习（强化成功的表达模式）

        不是拼接概念名，而是学习用户的表达方式。
        """
        # 1. SNN 处理输入
        result = self.process_input(user_msg)
        
        # 2. 从对话记忆中寻找相似的过去对话
        if self.conversation_history:
            # 找到最相似的过去输入
            best_match = None
            best_similarity = -1
            
            for past_input, past_output in self.conversation_history:
                # 简单相似度：共同概念数
                past_concepts, _, _ = self.scan_read(past_input)
                overlap = len(set(concepts) & set(past_concepts))
                if overlap > best_similarity:
                    best_similarity = overlap
                    best_match = (past_input, past_output)
            
            # 如果找到相似对话，参考它的输出
            if best_match and best_similarity > 0:
                _, past_output = best_match
                # 用过去输出作为基础，但加入当前输入的元素
                return self._adapt_past_response(user_msg, past_output, concepts)
        
        # 3. 如果没有参考，从用户输入中学习
        return self._learn_from_input(user_msg, concepts)

    def _adapt_past_response(self, user_msg, past_response, current_concepts):
        """
        适应过去的回复。

        不是复制，而是理解过去的回复模式，然后用当前输入生成类似表达。
        """
        # 分析过去回复的结构
        past_words = past_response.split()
        
        # 用当前输入中的词替换过去回复中的词
        # 这样回复会包含当前输入的元素，但保持过去的表达模式
        result_words = []
        input_words = user_msg.split()
        
        # 找到输入中的关键词（非停用词）
        stopwords = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那', '有', '和', '与', '或', '但', '而', '就', '都', '也', '还', '只', '不', '很', '个', '一', '二', '三', '四', '五'}
        input_keywords = [w for w in input_words if w not in stopwords and len(w) > 1]
        
        for word in past_words:
            if word in stopwords:
                # 停用词保持不变
                result_words.append(word)
            elif word in self.seed.concepts:
                # 概念词：尝试用当前输入的关键词替换
                if input_keywords:
                    # 找到与当前概念最相关的输入关键词
                    replacement = input_keywords[0]
                    input_keywords = input_keywords[1:]  # 消耗掉
                    result_words.append(replacement)
                else:
                    result_words.append(word)
            else:
                result_words.append(word)
        
        # 如果还有剩余关键词，添加到末尾
        for kw in input_keywords[:2]:
            if kw not in result_words:
                result_words.append(kw)
        
        return ' '.join(result_words[:7])  # 最多7个词

    def _learn_from_input(self, user_msg, concepts):
        """
        从用户输入中学习。

        不是拼接概念名，而是尝试用用户的方式表达。
        """
        # 从用户输入中提取关键词
        input_words = user_msg.split()
        
        # 找到关键词（非停用词）
        stopwords = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那', '有', '和', '与', '或', '但', '而', '就', '都', '也', '还', '只', '不', '很', '个', '一', '二', '三', '四', '五'}
        keywords = [w for w in input_words if w not in stopwords and len(w) > 1]
        
        # 如果有关键词，用它们生成回复
        if keywords:
            # 选择最相关的关键词
            # 这里简单地选择前几个，但未来可以基于SNN的激活来选择
            selected = keywords[:3]
            
            # 尝试用这些关键词生成一个回复
            # 这是学习的起点，不是最终答案
            return ' '.join(selected)
        
        # 如果没有关键词，用SNN的状态
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        # 找到变化最大的维度
        if not hasattr(self, '_baseline_tv') or self._baseline_tv is None:
            self._baseline_tv = tv.copy()
        
        change = tv - self._baseline_tv
        self._baseline_tv = tv.copy()
        
        # 用维度变化量选择概念
        top_dims = np.argsort(np.abs(change))[-3:]
        
        selected = []
        for dim in top_dims:
            # 找到与该维度最相关的概念
            for name, concept in self.seed.concepts.items():
                if hasattr(concept, 'semantic_dim') and concept.semantic_dim == dim:
                    selected.append(name)
                    break
        
        return ' '.join(selected) if selected else '……'

    def _emerge_from_thought_vector(self):
        """
        从 thought_vector 直接涌现语言。

        这是真正的涌现：SNN 的状态直接映射到概念。
        """
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)

        # 用变化量激活概念（和 express 系统一样）
        if not hasattr(self, '_baseline_tv') or self._baseline_tv is None:
            self._baseline_tv = tv.copy()

        change = tv - self._baseline_tv
        self._baseline_tv = tv.copy()

        # 按变化量排序
        region_size = len(change) // 10
        region_changes = []
        for i in range(10):
            start = i * region_size
            end = min((i + 1) * region_size, len(change))
            region_changes.append(np.sum(np.abs(change[start:end])))

        # 找到变化最大的区域
        sorted_regions = np.argsort(region_changes)[::-1]

        # 用区域变化量激活对应概念
        activated = []
        for ri in sorted_regions[:3]:
            # 找到该区域最活跃的概念
            for name, concept in self.seed.concepts.items():
                if hasattr(concept, 'region') and concept.region == ri:
                    if concept.activation > 0.1:
                        activated.append(name)
                        break

        if activated:
            return ' '.join(activated[:3])

        # 最后的后备：返回最活跃的概念
        active = [(n, c.activation) for n, c in self.seed.concepts.items() if c.activation > 0.1]
        if active:
            active.sort(key=lambda x: x[1], reverse=True)
            return ' '.join([n for n, _ in active[:3]])

        return '……'

    def _extract_words(self, text):
        """
        从中文文本中提取有意义的词。

        改进策略：
        1. 优先匹配已有概念和 CONCEPTS
        2. 双向扫描 + 严格过滤
        3. 只保留高质量词（2-4字）
        """
        import re
        # 去掉标点和空格
        clean = re.sub(r'[，。！？、；：""''（）《》\s\t\n]', '', text)
        if len(clean) < 2:
            return []

        words = set()

        # 策略1：提取 CONCEPTS 列表中出现在文本中的词
        for concept in CONCEPTS:
            if concept in clean and len(concept) >= 2:
                words.add(concept)

        # 策略2：提取已有概念中出现在文本中的词
        for concept_name in self.seed.concepts:
            if concept_name in clean and len(concept_name) >= 2:
                words.add(concept_name)

        # 策略3：双向扫描提取新词（更严格的过滤）
        # 常见虚词/功能字（不能作为词的首尾）
        func_chars = set('的了在是我你他她它们这那得着过吗呢吧啊呀哦嘛一二三四五六七八九十')
        # 常见动词/形容词起始字（更可能是真词的开头）
        good_start_chars = set('喜想看听吃走跑跳飞说读写学知觉感思爱恨怕惊美丽漂甜冷热')
        # 常见名词/形容词结尾字
        good_end_chars = set('天气水火风雪花草树木鸟鱼虫山河湖海音乐色彩光明暗热冷甜苦累')

        for length in [2, 3]:
            for i in range(len(clean) - length + 1):
                w = clean[i:i+length]
                # 过滤规则1：首尾不能是虚词
                if w[0] in func_chars or w[-1] in func_chars:
                    continue
                # 过滤规则2：不能是重复字（如"天天"）
                if len(set(w)) == 1:
                    continue
                # 过滤规则3：不能包含已在 words 中的词
                if any(w in existing or existing in w for existing in words if len(existing) >= 2):
                    continue
                # 过滤规则4：2字词必须有至少一个"好"字
                if length == 2:
                    if w[0] not in good_start_chars and w[-1] not in good_end_chars:
                        continue
                words.add(w)

        # 按长度排序（优先短词），取前 6 个
        result = sorted(words, key=len)[:6]
        return result

    def handle_user_message(self, msg):
        """
        处理一条用户消息。
        扫读 → 学词 → 注入概念 → 生成回复 → 记录对话 → 学习
        """
        text = msg.get('text', '')
        if not text:
            return

        # 特殊命令
        if text.startswith('/'):
            self._handle_chat_command(text)
            return

        # 扫读
        concepts, emotion_type, importance = self.scan_read(text)

        # 从用户消息中学习新词（扩展概念树！）
        new_words = self._extract_words(text)
        for word in new_words:
            self.seed.learn_word(word)

        # 将用户消息的概念注入种子（促进生长！）
        if concepts:
            self.seed.tick(
                poet_state={'concepts': concepts},
                accountant_state={'concepts': concepts},
                emotion=emotion_type,
                phi=float(self.brain.read_consciousness().phi)
            )

        # 同时将消息注入大脑（刺激 SNN）
        self.brain.inject_text(text)
        for _ in range(5):
            r = self.homeostasis.adjust_reward(0.001)
            self.brain.step(r)

        # 记录用户消息
        self.chat_say(f"[收到] {text}", msg_type='system')

        # 生成回复（始终回复！）
        should_reply, response = self.decide_respond(text, concepts, importance)

        if should_reply and response:
            self.chat_say(response, msg_type='reply')
            # 记录对话历史
            self.conversation_history.append((text, response))
            self.last_user_keywords = concepts
            
            # 从对话中学习：强化输入概念和输出概念之间的连接
            self._learn_from_conversation(text, response, concepts)
        else:
            # 不应该到这里（decide_respond 应该始终返回 True）
            self.chat_say("……", msg_type='reply')
            self.conversation_history.append((text, "……"))

    def _learn_from_conversation(self, user_msg, response, input_concepts):
        """
        从对话中学习。

        原理：像婴儿学说话，不是被教语法，而是通过模仿和反馈学会表达。
        - 输入概念和输出概念之间的连接被强化
        - 经常一起出现的概念变得更相关
        - 从对话模式中学习表达方式
        """
        # 1. 解析回复中的概念
        response_words = response.split()
        response_concepts = [w for w in response_words if w in self.seed.concepts]
        
        # 2. 强化输入概念和输出概念之间的连接
        for ic in input_concepts:
            if ic in self.seed.concepts:
                for rc in response_concepts:
                    if rc in self.seed.concepts and ic != rc:
                        # 强化连接
                        self.seed.concepts[ic].connect(rc, 0.1)
                        self.seed.concepts[rc].connect(ic, 0.1)
        
        # 3. 强化回复中概念之间的连接
        for i, c1 in enumerate(response_concepts):
            if c1 in self.seed.concepts:
                for c2 in response_concepts[i+1:]:
                    if c2 in self.seed.concepts and c1 != c2:
                        self.seed.concepts[c1].connect(c2, 0.05)
        
        # 4. 记录成功的对话模式
        if len(response_concepts) > 0:
            # 存储对话对，供未来参考
            if not hasattr(self, '_conversation_patterns'):
                self._conversation_patterns = []
            self._conversation_patterns.append({
                'input': user_msg,
                'output': response,
                'input_concepts': input_concepts,
                'output_concepts': response_concepts,
                'timestamp': time.time()
            })
            # 只保留最近100个模式
            if len(self._conversation_patterns) > 100:
                self._conversation_patterns = self._conversation_patterns[-100:]

    def _handle_chat_command(self, text):
        """处理聊天命令"""
        cmd = text.strip().lower()
        if cmd == '/status':
            ss = self.seed.get_status()
            self.chat_say(
                f"概念={ss['concepts']}/{ss['max_concepts']} "
                f"分叉={ss['total_forks']} 融合={ss['total_fusions']} "
                f"惊讶={ss['surprise_ema']:.3f} 节拍={self.tick_interval:.1f}s",
                msg_type='status'
            )
        elif cmd == '/block':
            self.chat_blocked = True
            self.chat_say("[系统] 已被拉黑，停止主动发言", msg_type='system')
        elif cmd == '/unblock':
            self.chat_blocked = False
            self.chat_say("[系统] 已解除拉黑", msg_type='system')
        elif cmd == '/fast':
            self.tick_interval = max(1.0, self.tick_interval - 2.0)
            self.chat_say(f"[系统] 节拍加速: {self.tick_interval:.1f}s", msg_type='system')
        elif cmd == '/slow':
            self.tick_interval = min(30.0, self.tick_interval + 5.0)
            self.chat_say(f"[系统] 节拍减速: {self.tick_interval:.1f}s", msg_type='system')


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
        sna.run_curriculum(150)  # 减少训练轮次，加速启动

        # 启动种子
        sna.seed.start_observation()
        print("[SNA] 种子已激活，进入100步静默观察期...", flush=True)
        print(f"[SNA] {sna.genome.describe()}", flush=True)

        # 清空聊天日志
        for f in [sna.chat_log_path, '/tmp/sna_chat_status.json']:
            try:
                open(f, 'w').close()
            except:
                pass

        print("[SNA] 守护进程模式 | 生长+对话统一循环 | 动态节拍", flush=True)
        print("[SNA] 思维链驱动 | 内部对话 | 自我模型 | 循环检测 | 意识树种子", flush=True)
        print(f"[SNA] 聊天日志: {sna.chat_log_path}", flush=True)
        thought_count = 0
        recent_growth_action = None

        while True:
            try:
                # === 0. 检查用户消息 ===
                sna.check_inbox()
                has_messages = len(sna.chat_inbox) > 0

                # === 1. 处理用户消息（如果有） ===
                if has_messages:
                    msg = sna.chat_inbox.popleft()
                    sna.handle_user_message(msg)
                    # 处理消息后不 sleep，立即进入下一步

                # === 2. 生成自主思维 ===
                brain_state = {
                    'phi': float(sna.brain.read_consciousness().phi),
                    'emotion': sna._get_python_emotion_label(),
                    'thought_vector': list(sna.brain.read_thought_vector()),
                }
                thought = sna._generate_spontaneous_thought(brain_state)
                recent_growth_action = None

                if thought:
                    thought_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")

                    # 标记思维类型
                    chain_depth = sna.thought_chain.chain_depth
                    loops_broken = sna.thought_chain.broken_loops
                    dialogue_count = len(sna.internal_dialogue.dialogue_history)
                    hs = sna.homeostasis.get_status()

                    print(f"[{ts}] 思维#{thought_count} [链深{chain_depth} 循环{loops_broken} 对话{dialogue_count}]: "
                          f"'{thought}' Phi={brain_state['phi']:.3f} 情感={brain_state['emotion']} "
                          f"活动={hs['activity']:.3f} 稳态={hs['status']}", flush=True)

                    # 将自主思维注入脑中
                    sna.brain.inject_text(thought)
                    for _ in range(10):
                        r = sna.homeostasis.adjust_reward(0.001)
                        sna.brain.step(r)
                        cs_inner = sna.brain.read_consciousness()
                        sna.homeostasis.measure_activity(region_activities=cs_inner.region_activities)

                    # 更新知识图谱
                    features = sna.get_features(thought)
                    biased_probs = sna.accountant.classify(features) * sna.corpus_callosum.get_bias()
                    top_idx = np.argsort(biased_probs)[-3:][::-1]
                    detected = [CONCEPTS[ci] for ci in top_idx if ci < N_CONCEPTS]
                    sna.curiosity.update_knowledge(detected)

                    # 种子心跳
                    seed_result = sna.seed.tick(
                        poet_state={'concepts': detected},
                        accountant_state={'concepts': detected},
                        emotion=brain_state['emotion'],
                        phi=brain_state['phi']
                    )
                    if seed_result:
                        recent_growth_action = seed_result.get('action')

                    # 生长监控
                    sna.growth_monitor.record(phi=brain_state['phi'])

                    # === 3. 树的主动发言（内部思维外化） ===
                    # 情感强烈或思维有趣时，推送到聊天界面
                    if thought_count % 3 == 0:
                        # 每3步，把内部对话的关键内容推送给用户
                        if sna.internal_dialogue.last_exchange:
                            ex = sna.internal_dialogue.last_exchange
                            if ex['tension'] > 0.4:
                                sna.chat_say(
                                    f"{ex['poet'][:60]}",
                                    msg_type='initiative'
                                )

                    # 定期显示内稳态和胼胝体状态
                    if thought_count % 5 == 0:
                        cs_status = sna.corpus_callosum.get_status()
                        print(f"  [稳态] 活动={hs['activity']:.3f} 修正={hs['correction']:+.3f} 区间={hs['target']}", flush=True)
                        print(f"  [胼胝体] 目标={cs_status['current_goal']} 整合={cs_status['integrations']}次", flush=True)

                    # 定期显示内部对话
                    if sna.internal_dialogue.last_exchange and thought_count % 5 == 0:
                        ex = sna.internal_dialogue.last_exchange
                        print(f"  [对话] 诗人：'{ex['poet'][:40]}'", flush=True)
                        print(f"  [对话] 会计：'{ex['accountant'][:40]}'", flush=True)
                        print(f"  [对话] 张力：{ex['tension']:.3f}", flush=True)

                    # 定期显示自我叙述
                    if thought_count % 10 == 0:
                        print(f"  [自我] {sna.self_model.get_narrative()}", flush=True)

                    # 定期显示种子状态
                    if thought_count % 10 == 0:
                        ss = sna.seed.get_status()
                        print(f"  [种子] {ss['phase']} 概念={ss['concepts']} "
                              f"创建={ss['total_created']} 修剪={ss['total_pruned']} "
                              f"分叉={ss['total_forks']} 融合={ss['total_fusions']} "
                              f"惊讶={ss['surprise_ema']:.3f}", flush=True)
                        for entry in sna.seed.get_growth_log(3):
                            print(f"  [生长] {entry['message']}", flush=True)

                    # 定期显示生长健康度
                    if thought_count % 20 == 0:
                        summary = sna.growth_monitor.get_summary()
                        health = summary['health']
                        print(f"  [健康] {health['status']} ({health['score']}/100) "
                              f"速率={summary['growth_rate']:.2f}%/步 "
                              f"概念使用={summary['concept_usage']:.1f}%", flush=True)
                        for alert in summary['alerts']:
                            print(f"  [告警] [{alert['severity']}] {alert['message']}", flush=True)

                    # 定期保存生长报告
                    if thought_count % 50 == 0:
                        sna.growth_monitor.save_state('growth_state.json')
                        sna.seed.save_state('seed_state.json')
                        print(f"  [保存] growth_state.json, seed_state.json", flush=True)

                # === 4. 写入状态文件（供前端读取） ===
                if thought_count % 5 == 0:
                    try:
                        ss = sna.seed.get_status()
                        status = {
                            'thought_count': thought_count,
                            'phi': brain_state['phi'],
                            'emotion': brain_state['emotion'],
                            'tick_interval': sna.tick_interval,
                            'blocked': sna.chat_blocked,
                            'seed': {
                                'phase': ss['phase'],
                                'concepts': ss['concepts'],
                                'max_concepts': ss['max_concepts'],
                                'surprise_ema': ss['surprise_ema'],
                                'total_forks': ss['total_forks'],
                                'total_fusions': ss['total_fusions'],
                                'total_pruned': ss['total_pruned'],
                            },
                            'ts': time.time(),
                        }
                        with open('/tmp/sna_chat_status.json', 'w') as f:
                            sna._json.dump(status, f)
                    except:
                        pass

                # === 5. 动态节拍 ===
                surprise = sna.seed.surprise_ema
                interval = sna.compute_tick_interval(surprise, has_messages, recent_growth_action)
                sna.tick_interval = interval
                time.sleep(interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[SNA] 错误: {e}", flush=True)
                import traceback
                traceback.print_exc()
                time.sleep(5)
    else:
        sna.interactive()


if __name__ == '__main__':
    main()
