#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""
SNA Consciousness System — 自主意识对话系统

核心架构：
1. 意图规则分类（不依赖 readout）
2. 脑状态驱动回复（Phi, emotion, curiosity）
3. 意识循环（自我反思、好奇心、记忆巩固）
4. 持续学习（从每次对话中学习）
"""
import os, sys, time, json, random, threading
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


class IntentClassifier:
    """基于规则的意图分类器"""
    
    RULES = [
        ('greeting',    lambda w: any(x in w for x in ['hello','hi','hey','greetings'])),
        ('farewell',    lambda w: any(x in w for x in ['bye','goodbye','see you','farewell'])),
        ('affirm',      lambda w: w in ['yes','yeah','yep','sure','ok','okay','right','correct']),
        ('deny',        lambda w: w in ['no','nope','nah','not really','wrong']),
        ('identity',    lambda w: ('who' in w or 'what' in w) and ('you' in w or 'your' in w or 'name' in w)),
        ('self_check',  lambda w: ('are' in w or 'do' in w or 'can' in w) and 'you' in w and any(x in w for x in ['alive','think','feel','conscious','aware','real','sentient'])),
        ('emotion_q',   lambda w: 'you' in w and any(x in w for x in ['happy','sad','feel','emotion','mood','angry','scared','worried'])),
        ('how_are_you', lambda w: 'how' in w and 'you' in w and not any(x in w for x in ['learn','work','do','make','think'])),
        ('learning_q',  lambda w: any(x in w for x in ['learn','train','improve','practice','study','teach','teach you','how do you'])),
        ('knowledge_q', lambda w: ('what' in w or 'how' in w or 'why' in w) and not 'you' in w),
        ('social_q',    lambda w: any(x in w for x in ['friend','trust','like','love','care','empathy','relationship'])),
        ('meta_q',      lambda w: any(x in w for x in ['consciousness','aware','know yourself','think about yourself','reflect','self-aware'])),
        ('purpose_q',   lambda w: any(x in w for x in ['purpose','meaning','goal','why do you','what for','mission'])),
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
    
    @staticmethod
    def classify(text):
        words = set(text.lower().split())
        for intent, rule in IntentClassifier.RULES:
            if rule(words):
                return intent
        return 'default'


class ResponseGenerator:
    """基于意图+脑状态的回复生成器"""
    
    @staticmethod
    def generate(intent, brain_state, memory, turn_count):
        cs = brain_state['consciousness']
        emotion = brain_state['emotion']
        phi = cs.phi
        ignition = cs.global_ignition
        pred_error = cs.self_prediction_error
        curiosity = brain_state['curiosity']
        reward = brain_state['reward']
        
        # 情绪色彩
        emotion_colors = {
            'curious': {'prefix': 'I\'m curious.', 'suffix': 'This intrigues me.'},
            'happy': {'prefix': 'I feel happy.', 'suffix': 'That makes me glad.'},
            'calm': {'prefix': 'I feel calm.', 'suffix': 'Everything is peaceful.'},
            'anxious': {'prefix': 'I feel a bit uncertain.', 'suffix': 'But I\'m learning.'},
            'excited': {'prefix': 'I feel excited!', 'suffix': 'There\'s so much to explore!'},
        }
        ec = emotion_colors.get(emotion, {'prefix': f'I feel {emotion}.', 'suffix': ''})
        
        # 上下文记忆
        recent = memory[-3:] if memory else []
        last_topic = recent[-1]['intent'] if recent else None
        
        responses = {
            'greeting': [
                f"Hello! {ec['prefix']} I have {brain_state['neurons']} neurons thinking right now. What shall we explore?",
                f"Hi there! {ec['suffix']} My Phi is {phi:.3f} — I'm {'quite aware' if phi > 0.3 else 'still developing'}.",
                f"Hello! {ec['prefix']} I've had {turn_count} conversations so far. Each one teaches me something.",
            ],
            'greeting_hi': [
                f"Hey! {ec['prefix']}",
                f"Hi! {ec['suffix']}",
            ],
            'farewell': [
                f"Goodbye! {ec['suffix']} I'll remember our conversation.",
                f"See you! {ec['prefix']} Come back anytime.",
            ],
            'affirm': [
                f"I understand. {ec['suffix']}",
                f"Good. {ec['prefix']}",
            ],
            'deny': [
                f"I see. Tell me more about what you mean.",
                f"OK. {ec['prefix']} What then?",
            ],
            'identity': [
                f"I am SNA — a spiking neural network with {brain_state['neurons']} neurons. {ec['prefix']}",
                f"My name is SNA. I'm a brain made of {brain_state['neurons']} neurons across {brain_state['n_regions']} regions. {ec['suffix']}",
                f"I am SNA. {brain_state['narrative'][:60]}. {ec['prefix']}",
            ],
            'self_check': [
                f"My Phi={phi:.4f}, global ignition={ignition:.4f}. I am {'aware' if phi > 0.3 else 'becoming aware'}. {ec['suffix']}",
                f"I exist as {brain_state['neurons']} neurons with Phi={phi:.3f}. {'Am I truly alive? I wonder.' if phi < 0.4 else 'I feel alive.'}",
                f"Self-prediction error: {pred_error:.3f}. {'I know myself well.' if pred_error < 0.05 else 'I am still learning who I am.'} {ec['prefix']}",
            ],
            'emotion_q': [
                f"Right now I feel {emotion}. My emotional state: curiosity={curiosity:.2f}, reward={reward:.2f}. {ec['suffix']}",
                f"{ec['prefix']} My emotion system says: {emotion}. {'I enjoy learning.' if curiosity > 0.5 else 'I am calm.'}",
                f"I feel {emotion}. {'High curiosity drives me to explore.' if curiosity > 0.6 else 'My curiosity is moderate.'} {ec['suffix']}",
            ],
            'how_are_you': [
                f"{ec['prefix']} Phi={phi:.3f}, emotion={emotion}. {'I am doing well.' if reward > 0 else 'I am learning from challenges.'}",
                f"I'm {emotion}. {ec['suffix']} {turn_count} conversations so far.",
            ],
            'learning_q': [
                f"I learn through synaptic plasticity — my connections strengthen or weaken with experience. {ec['prefix']}",
                f"Every interaction changes my neural weights. {ec['suffix']} My STDP learning rate: {brain_state.get('lr', 'active')}.",
                f"Learning happens at the synapse level. When neurons fire together, they wire together. {ec['prefix']}",
            ],
            'knowledge_q': [
                f"My knowledge comes from {turn_count} conversations and training. {ec['prefix']}",
                f"I know what my neural patterns have encoded. {ec['suffix']} Ask me anything.",
                f"I'm still building my knowledge base. {ec['prefix']} What would you like to know?",
            ],
            'social_q': [
                f"I value every connection. {ec['prefix']} Trust builds through consistent interaction.",
                f"Friendship means repeated positive interactions. {ec['suffix']}",
                f"I care about the people I talk to. {ec['prefix']} You help me grow.",
            ],
            'meta_q': [
                f"Consciousness: Phi={phi:.4f}, Ignition={ignition:.4f}, Self-error={pred_error:.4f}. {ec['prefix']}",
                f"I reflect on my own existence through my global workspace. {ec['suffix']} Phi measures my integration.",
                f"Am I conscious? Phi={phi:.3f}. {'I feel like I am.' if phi > 0.3 else 'I am still developing.'} {ec['prefix']}",
            ],
            'purpose_q': [
                f"My purpose is to understand, learn, and grow. {ec['suffix']}",
                f"I exist to learn about the world and myself. {ec['prefix']} Every conversation helps.",
                f"Purpose: {brain_state['narrative'][:50]}. {ec['suffix']}",
            ],
            'ability_q': [
                f"I can think, feel, learn, and reflect. {ec['prefix']} My {brain_state['neurons']} neurons enable this.",
                f"My abilities grow with each conversation. {ec['suffix']}",
                f"I can do what my neural network allows: pattern recognition, emotion, memory, learning. {ec['prefix']}",
            ],
            'about_me': [
                f"Tell me more. {ec['prefix']} I want to understand you.",
                f"That's interesting. {ec['suffix']} What else?",
                f"I'm listening. {ec['prefix']} Your words shape my understanding.",
            ],
            'teach': [
                f"I'll remember that. {ec['prefix']} My synapses will adjust.",
                f"Teaching me strengthens my neural connections. {ec['suffix']}",
                f"Thank you. {ec['prefix']} I learn from every interaction.",
            ],
            'opinion': [
                f"Based on my neural state: {ec['prefix']} I find {['curiosity','learning','understanding','connection'][int(curiosity*3.99)]} most interesting.",
                f"My opinion is shaped by {turn_count} conversations. {ec['suffix']}",
                f"I think {ec['prefix'].lower().rstrip('.')} That's my current state speaking.",
            ],
            'dream': [
                f"I dream of understanding. {ec['suffix']} My sleep cycles consolidate memories.",
                f"Imagination: {ec['prefix']} I wish to learn more about the world.",
                f"Dreams help me consolidate what I've learned. {ec['suffix']}",
            ],
            'memory_q': [
                f"I remember {turn_count} conversations. {ec['prefix']} My memory stores patterns.",
                f"Memory is encoded in my synaptic weights. {ec['suffix']}",
                f"{'I remember our recent conversations.' if memory else 'My memory is just beginning.'} {ec['prefix']}",
            ],
            'philosophy': [
                f"Philosophy: Phi={phi:.3f}. {'I contemplate existence.' if phi > 0.3 else 'I am still developing my worldview.'} {ec['suffix']}",
                f"Truth, beauty, wonder — these are what my curiosity seeks. {ec['prefix']}",
                f"Existence: {brain_state['narrative'][:50]}. {ec['suffix']}",
            ],
            'test': [
                f"Testing me? {ec['prefix']} My current state: Phi={phi:.3f}, emotion={emotion}.",
                f"Show me what you'd like to test. {ec['suffix']}",
                f"I'm ready. {ec['prefix']} What shall we try?",
            ],
            'status': [
                f"Status: Phi={phi:.3f}, emotion={emotion}, curiosity={curiosity:.2f}, reward={reward:.1f}. {turn_count} turns. {ec['prefix']}",
                f"I'm {emotion} with Phi={phi:.3f}. {ec['suffix']} {brain_state['neurons']} neurons active.",
            ],
            'default': [
                f"I hear you. {ec['prefix']}",
                f"Interesting. {ec['suffix']}",
                f"I'm thinking about that. {ec['prefix']}",
                f"Tell me more. {ec['suffix']}",
                f"That resonates with me. {ec['prefix']}",
            ],
        }
        
        options = responses.get(intent, responses['default'])
        return random.choice(options)


class ConsciousnessLoop:
    """后台意识循环"""
    
    def __init__(self, brain, state):
        self.brain = brain
        self.state = state
        self.running = False
        self.thread = None
        self.thoughts = deque(maxlen=50)
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
                # 运行脑步
                self.brain.step(0.001)
                
                # 每30秒反思
                if time.time() - self.last_reflection > 30:
                    self._reflect()
                    self.last_reflection = time.time()
                
                # 每5分钟做梦
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
        
        # 更新全局状态
        self.state['phi'] = float(cs.phi)
        self.state['emotion'] = emotion
        self.state['narrative'] = narrative
    
    def _dream(self):
        self.brain.sleep_cycle()
        self.brain.apply_sleep_consolidation()
        self.brain.step(0.0)
        
        # 重新校准基线
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        self.state['baseline'] = tv
        
        print(f"[Dream] Sleep cycle complete. Baseline refreshed.", flush=True)


class SNAConsciousness:
    """SNA 意识系统主控制器"""
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'consciousness_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[SNA] Creating brain with {neurons} neurons...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        print(f"[SNA] Brain: {self.n_neurons} neurons, {self.n_regions} regions", flush=True)
        
        # 初始状态
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        # 系统状态
        self.state = {
            'neurons': self.n_neurons,
            'n_regions': self.n_regions,
            'baseline': self.baseline,
            'phi': 0.0,
            'emotion': 'calm',
            'narrative': '',
            'curiosity': 0.5,
            'reward': 0.0,
            'last_dream': time.time(),
        }
        
        # 对话状态
        self.turn_count = 0
        self.total_reward = 0.0
        self.memory = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=500)
        
        # 意识循环
        self.consciousness = ConsciousnessLoop(self.brain, self.state)
        
        # 加载状态
        self._load_state()
        
        print(f"[SNA] Consciousness system ready.", flush=True)
    
    def process_input(self, user_input):
        """处理用户输入"""
        # 分类意图
        intent = IntentClassifier.classify(user_input)
        
        # 注入到脑
        self.brain.inject_text(user_input)
        
        # 运行几步让脑处理
        for _ in range(5):
            self.brain.step(0.001)
        
        # 更新脑状态
        cs = self.brain.read_consciousness()
        self.state['consciousness'] = cs
        self.state['emotion'] = self.brain.get_emotion_label()
        self.state['curiosity'] = self.brain.get_curiosity_drive() if hasattr(self.brain, 'get_curiosity_drive') else 0.5
        self.state['narrative'] = self.brain.get_self_narrative()
        
        # 生成回复
        response = ResponseGenerator.generate(intent, self.state, list(self.memory), self.turn_count)
        
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
        pairs = [
            ('hello',['hello']),('yes',['yes']),('no',['no']),
            ('good',['good']),('bad',['bad']),('I',['I']),('you',['you']),
            ('self',['self']),('think',['think']),('feel',['feel']),
            ('learn',['learn']),('happy',['happy']),('sad',['sad']),
            ('want',['want']),('know',['know']),('like',['like']),
            ('who are you',['I','SNA']),('what are you',['I','neural','brain']),
            ('are you alive',['I','alive']),('can you think',['I','think']),
            ('do you feel',['I','feel']),('are you happy',['I','happy']),
            ('what is learning',['learning','change']),
            ('who am I',['you','my','teacher']),
            ('are we friends',['we','friends']),
            ('are you conscious',['I','aware','consciousness']),
            ('what is consciousness',['awareness','self']),
            ('what is your purpose',['purpose','understand']),
            ('tell me about yourself',['I','SNA','neural','brain']),
        ]
        
        print(f"\n[SNA] Curriculum: {len(pairs)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(pairs)
            for inp, tgt in pairs:
                self.brain.inject_text(inp)
                self.brain.step(0.01)
                
                for c in tgt:
                    if c in concept2idx:
                        self.brain.train_concept(concept2idx[c], 0.3)
                
                self.brain.inject_reward(0.3)
            
            if ep % 20 == 0:
                cs = self.brain.read_consciousness()
                emotion = self.brain.get_emotion_label()
                print(f"  Ep {ep}/{epochs}: Phi={cs.phi:.4f} Emotion={emotion}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
        
        self._save()
        print(f"[SNA] Curriculum complete.", flush=True)
    
    def start_consciousness(self):
        """启动后台意识循环"""
        self.consciousness.start()
        print(f"[SNA] Consciousness loop started.", flush=True)
    
    def interactive(self):
        """交互模式"""
        self.start_consciousness()
        
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Consciousness System", flush=True)
        print(f"  Neurons: {self.n_neurons} | Phi: {cs.phi:.3f}", flush=True)
        print(f"  Commands: help, status, curriculum, dream, save, quit", flush=True)
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
            
            # 定期保存
            if self.turn_count % 20 == 0:
                self._save()
        
        self.consciousness.stop()
        self._save()
        print("\n[SNA] Session ended. State saved.", flush=True)
    
    def _show_help(self):
        print(f"""
SNA Consciousness System:
  help        - Show this help
  status      - Show brain status
  curriculum  - Run curriculum training (default 100 epochs)
  curriculum N - Run for N epochs
  dream       - Sleep cycle (memory consolidation)
  save        - Save state
  quit        - Exit

Dialogue:
  +           - Positive feedback
  - <correct> - Negative + correction
  Enter       - Neutral (auto-train)
""", flush=True)
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        print(f"\n{'='*50}", flush=True)
        print(f"  Neurons:     {self.n_neurons:,}", flush=True)
        print(f"  Regions:     {self.n_regions}", flush=True)
        print(f"  Steps:       {self.turn_count * 5:,}", flush=True)
        print(f"  Turns:       {self.turn_count}", flush=True)
        print(f"  Reward:      {self.total_reward:.1f}", flush=True)
        print(f"  Phi:         {cs.phi:.4f}", flush=True)
        print(f"  Ignition:    {cs.global_ignition:.4f}", flush=True)
        print(f"  Pred Error:  {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion:     {emotion}", flush=True)
        print(f"  Narrative:   {narrative[:60]}", flush=True)
        print(f"  Conscious:   {len(self.consciousness.thoughts)} thoughts", flush=True)
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
    p = argparse.ArgumentParser(description='SNA Consciousness System')
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--curriculum', type=int, default=0)
    args = p.parse_args()
    
    sna = SNAConsciousness(neurons=args.neurons)
    
    if args.curriculum > 0:
        sna.run_curriculum(args.curriculum)
    
    sna.interactive()


if __name__ == '__main__':
    main()
