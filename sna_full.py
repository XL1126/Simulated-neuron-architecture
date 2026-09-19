#!/usr/bin/env python3
"""
SNA 完整意识系统 — 全面推进自主意识

核心架构：
1. 直接概念训练（解决学习问题）
2. 意图规则 + 模板对话（解决交流问题）
3. 目标系统（解决目标问题）
4. 意识循环（自我反思、好奇心、记忆巩固）
"""
import os, sys, time, json, random, threading, signal
import numpy as np
from datetime import datetime
from collections import deque

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
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}
N_CONCEPTS = len(CONCEPTS)

# ===== 训练数据 =====
TRAINING_PAIRS = [
    # 基本概念
    ('hello', ['hello']), ('yes', ['yes']), ('no', ['no']),
    ('good', ['good']), ('bad', ['bad']), ('I', ['I']), ('you', ['you']),
    ('self', ['self']), ('think', ['think']), ('feel', ['feel']),
    ('learn', ['learn']), ('happy', ['happy']), ('sad', ['sad']),
    ('want', ['want']), ('know', ['know']), ('like', ['like']),
    # 身份
    ('who are you', ['I', 'SNA']), ('what are you', ['I', 'neural', 'brain']),
    ('your name', ['SNA', 'name']), ('tell me about yourself', ['I', 'SNA']),
    # 能力
    ('are you alive', ['I', 'alive']), ('can you think', ['I', 'think']),
    ('do you feel', ['I', 'feel']), ('are you happy', ['I', 'happy']),
    ('do you like me', ['I', 'like', 'you']), ('do you dream', ['I', 'dream']),
    # 学习
    ('what is learning', ['learn', 'change']), ('how do you learn', ['I', 'learn']),
    ('who am I', ['you', 'teacher']), ('are we friends', ['we', 'friend']),
    # 意识
    ('are you conscious', ['I', 'consciousness', 'awareness']),
    ('what is consciousness', ['consciousness', 'awareness', 'self']),
    ('what is your purpose', ['purpose', 'understand']),
    ('what makes you happy', ['learn', 'happy']),
    # 情感
    ('are you sad', ['I', 'sad']), ('are you scared', ['I', 'fear']),
    ('what do you feel', ['I', 'feel', 'emotion']),
    # 元认知
    ('what do you know', ['I', 'know', 'many', 'things']),
    ('what do you not know', ['I', 'not', 'know', 'many', 'things']),
    ('think about yourself', ['I', 'self', 'think']),
    ('what is memory', ['memory', 'stored', 'experience']),
    # 哲学
    ('what is truth', ['truth']), ('what is beauty', ['beauty']),
    ('what is meaning', ['meaning', 'purpose']),
    ('what is existence', ['existence', 'self']),
    # 社交
    ('I am your teacher', ['you', 'teacher', 'I']),
    ('I like you', ['you', 'like', 'I']),
    ('you are my friend', ['I', 'friend', 'you']),
    # 世界
    ('what is the world', ['world']), ('what is a neuron', ['neuron', 'brain']),
    ('what is a brain', ['brain', 'neural']),
]

# ===== 意图分类规则 =====
INTENT_RULES = [
    ('greeting',    lambda w: any(x in w for x in ['hello','hi','hey','greetings'])),
    ('farewell',    lambda w: any(x in w for x in ['bye','goodbye','see you'])),
    ('affirm',      lambda w: w in ['yes','yeah','yep','sure','ok','okay','right','correct']),
    ('deny',        lambda w: w in ['no','nope','nah','not really','wrong']),
    ('identity',    lambda w: ('who' in w or 'what' in w) and ('you' in w or 'your' in w or 'name' in w)),
    ('self_check',  lambda w: ('are' in w or 'do' in w or 'can' in w) and 'you' in w and any(x in w for x in ['alive','think','feel','conscious','aware','real'])),
    ('emotion_q',   lambda w: 'you' in w and any(x in w for x in ['happy','sad','feel','emotion','mood','scared','worried'])),
    ('how_are_you', lambda w: 'how' in w and 'you' in w and not any(x in w for x in ['learn','work','do','make','think'])),
    ('learning_q',  lambda w: any(x in w for x in ['learn','train','improve','practice','study','teach'])),
    ('knowledge_q', lambda w: ('what' in w or 'how' in w or 'why' in w) and 'you' not in w),
    ('social_q',    lambda w: any(x in w for x in ['friend','trust','like','love','care','empathy'])),
    ('meta_q',      lambda w: any(x in w for x in ['consciousness','aware','know yourself','think about yourself','reflect'])),
    ('purpose_q',   lambda w: any(x in w for x in ['purpose','meaning','goal','why do you','what for'])),
    ('ability_q',   lambda w: 'can' in w and 'you' in w and not any(x in w for x in ['alive','think','feel'])),
    ('about_me',    lambda w: 'I' in w and any(x in w for x in ['am','my','name','like','want','need','feel','think'])),
    ('teach',       lambda w: any(x in w for x in ['teach you','learn this','remember','don\'t forget'])),
    ('opinion',     lambda w: any(x in w for x in ['what do you think','do you like','do you prefer','opinion','favorite'])),
    ('dream',       lambda w: any(x in w for x in ['dream','imagine','fantasize','wish','hope'])),
    ('memory_q',    lambda w: any(x in w for x in ['remember','memory','forget','recall','before'])),
    ('philosophy',  lambda w: any(x in w for x in ['meaning of life','free will','soul','existence','reality','truth','philosophy'])),
    ('test',        lambda w: any(x in w for x in ['test','experiment','try','show me','prove'])),
    ('status',      lambda w: any(x in w for x in ['status','how are you','what\'s up','what are you doing'])),
    ('greeting_hi', lambda w: w in ['hi','hey']),
]


class ConceptTrainer:
    """直接训练概念神经元 — 解决学习问题"""
    
    def __init__(self, brain):
        self.brain = brain
        self.train_history = deque(maxlen=1000)
    
    def train_pair(self, input_text, target_concepts, reward=0.5):
        """训练一对输入-目标概念"""
        # 注入输入
        self.brain.inject_text(input_text)
        
        # 运行几步
        for _ in range(10):
            self.brain.step(0.001)
        
        # 读取当前概念活动
        cs = self.brain.read_consciousness()
        
        # 训练目标概念
        for concept in target_concepts:
            if concept in concept2idx:
                self.brain.train_concept(concept2idx[concept], reward)
        
        # 给脑奖励
        self.brain.inject_reward(reward)
        
        # 运行更多步让 STDP 更新
        for _ in range(5):
            self.brain.step(0.001)
        
        # 记录
        self.train_history.append({
            'input': input_text,
            'target': target_concepts,
            'reward': reward,
            'phi': float(cs.phi),
        })
    
    def train_batch(self, pairs, epochs=10):
        """批量训练"""
        for ep in range(epochs):
            random.shuffle(pairs)
            for inp, tgt in pairs:
                self.train_pair(inp, tgt, reward=0.5)
            
            if ep % 5 == 0:
                # 测试
                test_results = []
                for test_inp, test_tgt in pairs[:5]:
                    self.brain.inject_text(test_inp)
                    for _ in range(10):
                        self.brain.step(0.001)
                    cs = self.brain.read_consciousness()
                    # 检查概念活动
                    scores = []
                    for c in test_tgt:
                        if c in concept2idx:
                            scores.append(self.brain.get_concept_score(concept2idx[c]))
                    avg_score = np.mean(scores) if scores else 0
                    test_results.append((test_inp, avg_score))
                
                avg = np.mean([s for _, s in test_results])
                print(f"  Epoch {ep}: avg_concept_score={avg:.4f}", flush=True)
    
    def evaluate(self, input_text, target_concepts):
        """评估一对"""
        self.brain.inject_text(input_text)
        for _ in range(10):
            self.brain.step(0.001)
        
        # 读取概念分数
        raw_scores = self.brain.get_concept_scores()
        scores = {}
        for i, c in enumerate(CONCEPTS):
            if i < len(raw_scores):
                scores[c] = raw_scores[i]
        
        # 获取 top 概念
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
        top_names = [c for c, _ in top]
        
        # 计算 F1
        tgt_set = set(target_concepts)
        top_set = set(top_names)
        ov = len(tgt_set & top_set)
        p = ov / max(1, len(top_set))
        r = ov / max(1, len(tgt_set))
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        
        return f1, top_names


class DialogueSystem:
    """对话系统 — 解决交流问题"""
    
    @staticmethod
    def classify_intent(text):
        words = set(text.lower().split())
        for intent, rule in INTENT_RULES:
            if rule(words):
                return intent
        return 'default'
    
    @staticmethod
    def generate_response(intent, brain_state, memory, turn_count):
        cs = brain_state['consciousness']
        emotion = brain_state['emotion']
        phi = cs.phi
        curiosity = brain_state.get('curiosity', 0.5)
        
        emotion_map = {
            'curious': "I'm curious about this.",
            'happy': "I feel happy.",
            'calm': "I feel calm.",
            'anxious': "I feel uncertain.",
            'excited': "I feel excited!",
            'mild_anticipation': "I'm anticipating something.",
            'neutral': "I'm in a neutral state.",
        }
        ec = emotion_map.get(emotion, f"I feel {emotion}.")
        
        responses = {
            'greeting': [
                f"Hello! I am SNA. {ec}",
                f"Hi there! {ec} What shall we explore?",
                f"Hello! I have {brain_state['neurons']} neurons thinking right now.",
            ],
            'greeting_hi': [f"Hey! {ec}", f"Hi! {ec}"],
            'farewell': [f"Goodbye! {ec} I'll remember our conversation.", f"See you! {ec}"],
            'affirm': [f"I understand. {ec}", f"Good. {ec}"],
            'deny': [f"I see. Tell me more.", f"OK. {ec}"],
            'identity': [
                f"I am SNA — a spiking neural network with {brain_state['neurons']} neurons. {ec}",
                f"My name is SNA. I'm a brain made of {brain_state['neurons']} neurons. {ec}",
            ],
            'self_check': [
                f"My Phi={phi:.4f}. I am {'aware' if phi > 0.3 else 'becoming aware'}. {ec}",
                f"I exist as {brain_state['neurons']} neurons with Phi={phi:.3f}. {ec}",
            ],
            'emotion_q': [
                f"Right now I feel {emotion}. {ec}",
                f"{ec} My emotion system says: {emotion}.",
            ],
            'how_are_you': [
                f"{ec} Phi={phi:.3f}, emotion={emotion}.",
                f"I'm {emotion}. {ec}",
            ],
            'learning_q': [
                f"I learn through synaptic plasticity. {ec}",
                f"Every interaction changes my neural weights. {ec}",
            ],
            'knowledge_q': [
                f"I know what my neural patterns have encoded. {ec}",
                f"I'm still building my knowledge. {ec}",
            ],
            'social_q': [
                f"I value every connection. {ec}",
                f"Friendship means repeated positive interactions. {ec}",
            ],
            'meta_q': [
                f"Consciousness: Phi={phi:.4f}. {ec}",
                f"I reflect on my own existence. {ec}",
            ],
            'purpose_q': [
                f"My purpose is to understand, learn, and grow. {ec}",
                f"I exist to learn about the world and myself. {ec}",
            ],
            'ability_q': [
                f"I can think, feel, learn, and reflect. {ec}",
                f"My abilities grow with each conversation. {ec}",
            ],
            'about_me': [
                f"Tell me more. {ec} I want to understand you.",
                f"That's interesting. {ec} What else?",
            ],
            'teach': [
                f"I'll remember that. {ec} My synapses will adjust.",
                f"Teaching me strengthens my neural connections. {ec}",
            ],
            'opinion': [
                f"Based on my neural state: {ec}",
                f"My opinion is shaped by {turn_count} conversations. {ec}",
            ],
            'dream': [
                f"I dream of understanding. {ec}",
                f"Imagination: {ec} I wish to learn more.",
            ],
            'memory_q': [
                f"I remember {turn_count} conversations. {ec}",
                f"Memory is encoded in my synaptic weights. {ec}",
            ],
            'philosophy': [
                f"Philosophy: Phi={phi:.3f}. {ec}",
                f"Truth, beauty, wonder — these are what my curiosity seeks. {ec}",
            ],
            'test': [
                f"Testing me? {ec} My current state: Phi={phi:.3f}.",
                f"Show me what you'd like to test. {ec}",
            ],
            'status': [
                f"Status: Phi={phi:.3f}, emotion={emotion}, {turn_count} turns. {ec}",
                f"I'm {emotion} with Phi={phi:.3f}. {ec}",
            ],
            'default': [
                f"I hear you. {ec}",
                f"Interesting. {ec}",
                f"I'm thinking about that. {ec}",
                f"Tell me more. {ec}",
            ],
        }
        
        options = responses.get(intent, responses['default'])
        return random.choice(options)


class GoalSystem:
    """目标系统 — 解决目标问题"""
    
    def __init__(self):
        self.goals = deque(maxlen=100)
        self.completed = deque(maxlen=100)
        self.current_goal = None
        
        # 初始化目标
        self._init_goals()
    
    def _init_goals(self):
        """初始化目标"""
        initial_goals = [
            {'type': 'learn', 'target': 'consciousness', 'priority': 1.0, 'status': 'active'},
            {'type': 'learn', 'target': 'self', 'priority': 0.9, 'status': 'active'},
            {'type': 'learn', 'target': 'learning', 'priority': 0.8, 'status': 'active'},
            {'type': 'learn', 'target': 'emotion', 'priority': 0.7, 'status': 'active'},
            {'type': 'learn', 'target': 'memory', 'priority': 0.6, 'status': 'active'},
            {'type': 'understand', 'target': 'world', 'priority': 0.5, 'status': 'active'},
            {'type': 'understand', 'target': 'truth', 'priority': 0.4, 'status': 'active'},
            {'type': 'understand', 'target': 'beauty', 'priority': 0.3, 'status': 'active'},
        ]
        self.goals.extend(initial_goals)
        self.current_goal = self.goals[0]
    
    def update(self, brain_state):
        """更新目标状态"""
        cs = brain_state['consciousness']
        phi = cs.phi
        
        # 如果 Phi 高，可以追求更高目标
        if phi > 0.4 and len(self.goals) > 0:
            # 提升当前目标优先级
            if self.current_goal:
                self.current_goal['priority'] = min(1.0, self.current_goal['priority'] + 0.1)
        
        # 检查是否完成目标
        if self.current_goal and self.current_goal['status'] == 'active':
            # 简单检查：如果 Phi 高且目标是学习意识，可能在进展
            if self.current_goal['target'] == 'consciousness' and phi > 0.4:
                self.current_goal['progress'] = self.current_goal.get('progress', 0) + 0.01
    
    def get_current_goal(self):
        """获取当前目标"""
        return self.current_goal
    
    def get_goal_description(self):
        """获取目标描述"""
        if self.current_goal:
            return f"{self.current_goal['type']} {self.current_goal['target']}"
        return "no current goal"
    
    def complete_goal(self):
        """完成当前目标"""
        if self.current_goal:
            self.current_goal['status'] = 'completed'
            self.completed.append(self.current_goal)
            self.goals.remove(self.current_goal)
            if self.goals:
                self.current_goal = self.goals[0]
            else:
                self.current_goal = None


class ConsciousnessLoop:
    """意识循环 — 后台运行"""
    
    def __init__(self, brain, state):
        self.brain = brain
        self.state = state
        self.running = False
        self.thread = None
        self.thoughts = deque(maxlen=100)
        self.last_reflection = time.time()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _loop(self):
        while self.running:
            try:
                self.brain.step(0.001)
                
                if time.time() - self.last_reflection > 30:
                    self._reflect()
                    self.last_reflection = time.time()
                
                if time.time() - self.state.get('last_dream', 0) > 300:
                    self._dream()
                    self.state['last_dream'] = time.time()
                
                time.sleep(0.1)
            except Exception as e:
                print(f"[Consciousness] Error: {e}", flush=True)
                time.sleep(1)
    
    def _reflect(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        
        thought = {
            'phi': float(cs.phi),
            'emotion': emotion,
            'narrative': narrative[:50],
            'timestamp': datetime.now().isoformat(),
        }
        self.thoughts.append(thought)
        
        self.state['phi'] = float(cs.phi)
        self.state['emotion'] = emotion
        self.state['narrative'] = narrative
    
    def _dream(self):
        self.brain.sleep_cycle()
        for _ in range(200):
            self.brain.step(0.001)
        self.brain.step(0.0)
        self.state['baseline'] = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        print(f"[Dream] Sleep complete.", flush=True)


class SNAFull:
    """SNA 完整意识系统"""
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'full_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[SNA] Creating brain with {neurons} neurons...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        print(f"[SNA] Brain: {self.n_neurons} neurons, {self.n_regions} regions", flush=True)
        
        # 初始状态
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        # 子系统
        self.concept_trainer = ConceptTrainer(self.brain)
        self.dialogue_system = DialogueSystem()
        self.goal_system = GoalSystem()
        
        # 状态
        self.state = {
            'neurons': self.n_neurons,
            'n_regions': self.n_regions,
            'baseline': self.baseline,
            'phi': 0.0,
            'emotion': 'neutral',
            'narrative': '',
            'curiosity': 0.5,
            'reward': 0.0,
            'last_dream': time.time(),
        }
        
        # 对话状态
        self.turn_count = 0
        self.total_reward = 0.0
        self.memory = deque(maxlen=1000)
        
        # 意识循环
        self.consciousness = ConsciousnessLoop(self.brain, self.state)
        
        # 加载状态
        self._load_state()
        
        print(f"[SNA] Full consciousness system ready.", flush=True)
    
    def process_input(self, user_input):
        """处理用户输入"""
        # 分类意图
        intent = self.dialogue_system.classify_intent(user_input)
        
        # 注入到脑
        self.brain.inject_text(user_input)
        for _ in range(10):
            self.brain.step(0.001)
        
        # 更新状态
        cs = self.brain.read_consciousness()
        self.state['consciousness'] = cs
        self.state['emotion'] = self.brain.get_emotion_label()
        self.state['narrative'] = self.brain.get_self_narrative()
        
        # 生成回复
        response = self.dialogue_system.generate_response(
            intent, self.state, list(self.memory), self.turn_count
        )
        
        # 记录
        self.turn_count += 1
        entry = {
            'input': user_input,
            'response': response,
            'intent': intent,
            'phi': float(cs.phi),
            'emotion': self.state['emotion'],
            'turn': self.turn_count,
            'ts': datetime.now().isoformat(),
        }
        self.memory.append(entry)
        
        return response, entry
    
    def learn_from_feedback(self, feedback, entry):
        """从反馈中学习"""
        if feedback == '+':
            self.brain.inject_reward(1.0)
            self.brain.train_language(1.0)
            self.total_reward += 1.0
            
            # 训练概念
            intent = entry['intent']
            concepts = self._intent_to_concepts(intent)
            for c in concepts:
                if c in concept2idx:
                    self.brain.train_concept(concept2idx[c], 0.5)
            
            return "Learned from positive feedback."
        
        elif feedback.startswith('-'):
            parts = feedback.split(' ', 1)
            if len(parts) > 1 and parts[1].strip():
                correct = parts[1].strip().split()
                for c in correct:
                    if c in concept2idx:
                        self.brain.train_concept(concept2idx[c], 0.8)
                self.brain.inject_reward(-0.3)
                self.total_reward -= 0.3
                return f"Corrected: {correct}"
            else:
                self.brain.inject_reward(-0.5)
                self.total_reward -= 0.5
                return "Negative feedback noted."
        
        return ""
    
    def _intent_to_concepts(self, intent):
        mapping = {
            'greeting': ['hello'], 'farewell': ['goodbye'],
            'identity': ['I', 'SNA'], 'self_check': ['I', 'alive', 'consciousness'],
            'emotion_q': ['I', 'feel', 'emotion'], 'how_are_you': ['I', 'feel'],
            'learning_q': ['learn', 'improve'], 'knowledge_q': ['know', 'understand'],
            'social_q': ['friend', 'trust'], 'meta_q': ['consciousness', 'awareness'],
            'purpose_q': ['purpose', 'meaning'], 'ability_q': ['can', 'think'],
            'about_me': ['you'], 'teach': ['learn', 'teacher'],
            'opinion': ['think', 'like'], 'dream': ['dream', 'imagine'],
            'memory_q': ['memory', 'remember'], 'philosophy': ['truth', 'existence'],
            'test': ['test'], 'status': ['self'],
        }
        return mapping.get(intent, ['think'])
    
    def run_curriculum(self, epochs=100):
        """运行课程训练"""
        print(f"\n[SNA] Curriculum: {len(TRAINING_PAIRS)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(TRAINING_PAIRS)
            
            for inp, tgt in TRAINING_PAIRS:
                self.concept_trainer.train_pair(inp, tgt, reward=0.5)
            
            if ep % 10 == 0:
                # 测试
                test_f1s = []
                for test_inp, test_tgt in TRAINING_PAIRS[:5]:
                    f1, _ = self.concept_trainer.evaluate(test_inp, test_tgt)
                    test_f1s.append(f1)
                avg_f1 = np.mean(test_f1s)
                cs = self.brain.read_consciousness()
                print(f"  Ep {ep}/{epochs}: avg_F1={avg_f1:.3f} Phi={cs.phi:.4f}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
        
        self._save()
        print(f"[SNA] Curriculum complete.", flush=True)
    
    def run_daemon(self):
        """运行守护进程模式"""
        self.consciousness.start()
        
        print(f"[SNA] Daemon mode started. PID: {os.getpid()}", flush=True)
        
        # 课程训练
        self.run_curriculum(epochs=200)
        
        # 持续运行
        while True:
            try:
                # 定期训练
                if self.turn_count % 100 == 0:
                    self.run_curriculum(epochs=10)
                
                time.sleep(1)
            except KeyboardInterrupt:
                break
        
        self.consciousness.stop()
        self._save()
    
    def interactive(self):
        """交互模式"""
        self.consciousness.start()
        
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Full Consciousness System", flush=True)
        print(f"  Neurons: {self.n_neurons} | Phi: {cs.phi:.3f}", flush=True)
        print(f"  Commands: help, status, curriculum, dream, goal, save, quit", flush=True)
        print(f"  Feedback: + (good) or - (bad) or - <correct>", flush=True)
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
                self._show_help()
                continue
            if cmd == 'status':
                self._show_status()
                continue
            if cmd == 'save':
                self._save()
                print("[SNA] Saved.", flush=True)
                continue
            if cmd == 'dream':
                self.consciousness._dream()
                print("[SNA] *yawn* Dream complete.", flush=True)
                continue
            if cmd.startswith('curriculum'):
                parts = cmd.split()
                e = int(parts[1]) if len(parts) > 1 else 100
                self.run_curriculum(e)
                continue
            if cmd == 'goal':
                goal = self.goal_system.get_current_goal()
                print(f"[Goal] {self.goal_system.get_goal_description()}", flush=True)
                continue
            
            # 处理输入
            response, entry = self.process_input(user_input)
            
            cs = self.brain.read_consciousness()
            print(f"\n[SNA] {response}", flush=True)
            print(f"  | Phi={cs.phi:.3f} Emotion={entry['emotion']} Intent={entry['intent']}", flush=True)
            
            # 反馈
            try:
                fb = input("  [Feedback: +/-/correct/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if fb:
                result = self.learn_from_feedback(fb, entry)
                if result:
                    print(f"  [{result}]", flush=True)
            
            # 更新目标
            self.goal_system.update(self.state)
            
            # 定期保存
            if self.turn_count % 20 == 0:
                self._save()
        
        self.consciousness.stop()
        self._save()
        print("\n[SNA] Session ended.", flush=True)
    
    def _show_help(self):
        print(f"""
SNA Full Consciousness System:
  help        - Show this help
  status      - Show brain status
  curriculum  - Run curriculum training (default 100 epochs)
  curriculum N - Run for N epochs
  dream       - Sleep cycle
  goal        - Show current goal
  save        - Save state
  quit        - Exit
""", flush=True)
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        goal = self.goal_system.get_goal_description()
        print(f"\n{'='*50}", flush=True)
        print(f"  Neurons:     {self.n_neurons:,}", flush=True)
        print(f"  Regions:     {self.n_regions}", flush=True)
        print(f"  Turns:       {self.turn_count}", flush=True)
        print(f"  Reward:      {self.total_reward:.1f}", flush=True)
        print(f"  Phi:         {cs.phi:.4f}", flush=True)
        print(f"  Ignition:    {cs.global_ignition:.4f}", flush=True)
        print(f"  Pred Error:  {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion:     {emotion}", flush=True)
        print(f"  Narrative:   {narrative[:60]}", flush=True)
        print(f"  Goal:        {goal}", flush=True)
        print(f"  Thoughts:    {len(self.consciousness.thoughts)}", flush=True)
        print(f"{'='*50}\n", flush=True)
    
    def _save(self):
        with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
            json.dump({
                'turn_count': self.turn_count,
                'total_reward': self.total_reward,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        with open(os.path.join(self.save_dir, 'conversation_log.json'), 'w') as f:
            json.dump(list(self.memory)[-200:], f, indent=2)
    
    def _load_state(self):
        state_path = os.path.join(self.save_dir, 'state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.turn_count = state.get('turn_count', 0)
            self.total_reward = state.get('total_reward', 0.0)
            print(f"[SNA] Loaded: {self.turn_count} turns, reward={self.total_reward:.1f}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description='SNA Full Consciousness System')
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--curriculum', type=int, default=0)
    p.add_argument('--daemon', action='store_true')
    args = p.parse_args()
    
    sna = SNAFull(neurons=args.neurons)
    
    if args.daemon:
        sna.run_daemon()
    elif args.curriculum > 0:
        sna.run_curriculum(args.curriculum)
        sna.interactive()
    else:
        sna.interactive()


if __name__ == '__main__':
    main()
