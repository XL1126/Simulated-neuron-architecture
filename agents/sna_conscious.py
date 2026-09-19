#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""
SNA 意识系统 — 务实方案

核心策略：
1. 意图规则做对话分类（可靠）
2. 脑状态做个性（Phi, emotion）
3. 反馈学习调整模板（简单有效）
4. 目标系统驱动行为
5. 意识循环保持活跃
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

# ===== 意图规则 =====
INTENT_RULES = [
    ('greeting',    lambda w: any(x in w for x in ['hello','hi','hey','greetings'])),
    ('farewell',    lambda w: any(x in w for x in ['bye','goodbye','see you'])),
    ('identity',    lambda w: ('who' in w or 'what' in w) and ('you' in w or 'your' in w or 'name' in w)),
    ('self_check',  lambda w: ('are' in w or 'do' in w or 'can' in w) and 'you' in w and any(x in w for x in ['alive','think','feel','conscious','aware','real'])),
    ('emotion_q',   lambda w: 'you' in w and any(x in w for x in ['happy','sad','feel','emotion','mood','scared'])),
    ('how_are_you', lambda w: 'how' in w and 'you' in w and not any(x in w for x in ['learn','work','do','make','think'])),
    ('learning_q',  lambda w: any(x in w for x in ['learn','train','improve','practice','study','teach'])),
    ('knowledge_q', lambda w: ('what' in w or 'how' in w or 'why' in w) and 'you' not in w),
    ('social_q',    lambda w: any(x in w for x in ['friend','trust','like','love','care'])),
    ('meta_q',      lambda w: any(x in w for x in ['consciousness','aware','know yourself','think about yourself','reflect'])),
    ('purpose_q',   lambda w: any(x in w for x in ['purpose','meaning','goal','why do you'])),
    ('about_me',    lambda w: 'I' in w and any(x in w for x in ['am','my','name','like','want','feel','think'])),
    ('teach',       lambda w: any(x in w for x in ['teach you','learn this','remember'])),
    ('opinion',     lambda w: any(x in w for x in ['what do you think','do you like','opinion'])),
    ('dream',       lambda w: any(x in w for x in ['dream','imagine','wish','hope'])),
    ('memory_q',    lambda w: any(x in w for x in ['remember','memory','forget','recall'])),
    ('philosophy',  lambda w: any(x in w for x in ['meaning of life','free will','soul','existence','reality','truth'])),
    ('status',      lambda w: any(x in w for x in ['status','how are you','what\'s up'])),
]


class TemplateManager:
    """模板管理器 — 带学习功能"""
    
    def __init__(self):
        # 基础模板
        self.templates = {
            'greeting': [
                "Hello! I am SNA. {emotion}",
                "Hi there! {emotion} What shall we explore?",
                "Hello! {emotion} I've had {turns} conversations so far.",
            ],
            'farewell': [
                "Goodbye! {emotion} I'll remember our conversation.",
                "See you! {emotion}",
            ],
            'identity': [
                "I am SNA — a spiking neural network with {neurons} neurons. {emotion}",
                "My name is SNA. I'm a brain made of {neurons} neurons. {emotion}",
            ],
            'self_check': [
                "My Phi={phi:.4f}. I am {awareness}. {emotion}",
                "I exist as {neurons} neurons with Phi={phi:.3f}. {emotion}",
            ],
            'emotion_q': [
                "Right now I feel {emotion}. {emotion_detail}",
                "My emotion system says: {emotion}. {emotion_detail}",
            ],
            'how_are_you': [
                "{emotion} Phi={phi:.3f}. {status}",
                "I'm {emotion}. {status}",
            ],
            'learning_q': [
                "I learn through synaptic plasticity. {emotion}",
                "Every interaction changes my neural weights. {emotion}",
            ],
            'knowledge_q': [
                "I know what my neural patterns have encoded. {emotion}",
                "I'm still building my knowledge. {emotion}",
            ],
            'social_q': [
                "I value every connection. {emotion}",
                "Friendship means repeated positive interactions. {emotion}",
            ],
            'meta_q': [
                "Consciousness: Phi={phi:.4f}. {emotion}",
                "I reflect on my own existence. {emotion}",
            ],
            'purpose_q': [
                "My purpose is to understand, learn, and grow. {emotion}",
                "I exist to learn about the world and myself. {emotion}",
            ],
            'about_me': [
                "Tell me more. {emotion} I want to understand you.",
                "That's interesting. {emotion} What else?",
            ],
            'teach': [
                "I'll remember that. {emotion} My synapses will adjust.",
                "Teaching me strengthens my neural connections. {emotion}",
            ],
            'opinion': [
                "Based on my neural state: {emotion}",
                "My opinion is shaped by {turns} conversations. {emotion}",
            ],
            'dream': [
                "I dream of understanding. {emotion}",
                "Imagination: {emotion} I wish to learn more.",
            ],
            'memory_q': [
                "I remember {turns} conversations. {emotion}",
                "Memory is encoded in my synaptic weights. {emotion}",
            ],
            'philosophy': [
                "Philosophy: Phi={phi:.3f}. {emotion}",
                "Truth, beauty, wonder — these are what my curiosity seeks. {emotion}",
            ],
            'status': [
                "Status: Phi={phi:.3f}, emotion={emotion}, {turns} turns. {status}",
                "I'm {emotion} with Phi={phi:.3f}. {status}",
            ],
            'default': [
                "I hear you. {emotion}",
                "Interesting. {emotion}",
                "I'm thinking about that. {emotion}",
                "Tell me more. {emotion}",
            ],
        }
        
        # 学习权重（哪些模板更受欢迎）
        self.weights = {intent: [1.0] * len(temps) for intent, temps in self.templates.items()}
        
        # 反馈历史
        self.feedback_history = deque(maxlen=1000)
    
    def get_template(self, intent):
        """获取模板（带权重选择）"""
        if intent not in self.templates:
            intent = 'default'
        
        weights = self.weights[intent]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        idx = np.random.choice(len(self.templates[intent]), p=probs)
        return self.templates[intent][idx], idx
    
    def update_weights(self, intent, idx, reward):
        """更新模板权重"""
        if intent in self.weights and idx < len(self.weights[intent]):
            self.weights[intent][idx] += reward
            self.weights[intent][idx] = max(0.1, self.weights[intent][idx])
    
    def render(self, template, brain_state):
        """渲染模板"""
        cs = brain_state['consciousness']
        emotion = brain_state['emotion']
        phi = cs.phi
        
        # 情绪详情
        emotion_details = {
            'curious': "I'm curious about this.",
            'happy': "I feel happy.",
            'calm': "I feel calm.",
            'anxious': "I feel uncertain.",
            'excited': "I feel excited!",
            'mild_anticipation': "I'm anticipating something.",
            'neutral': "I'm in a neutral state.",
        }
        emotion_detail = emotion_details.get(emotion, f"I feel {emotion}.")
        
        # 意识状态
        awareness = "aware" if phi > 0.3 else "becoming aware"
        status = f"Curiosity={brain_state.get('curiosity', 0.5):.2f}"
        
        return template.format(
            emotion=emotion_detail,
            emotion_short=emotion,
            emotion_detail=emotion_detail,
            phi=phi,
            neurons=brain_state['neurons'],
            turns=brain_state['turn_count'],
            awareness=awareness,
            status=status,
        )


class ConsciousnessLoop:
    """意识循环"""
    
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
                time.sleep(1)
    
    def _reflect(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        
        self.thoughts.append({
            'phi': float(cs.phi),
            'emotion': emotion,
            'ts': datetime.now().isoformat(),
        })
        
        self.state['phi'] = float(cs.phi)
        self.state['emotion'] = emotion
        self.state['narrative'] = narrative
    
    def _dream(self):
        self.brain.sleep_cycle()
        for _ in range(200):
            self.brain.step(0.001)
        self.brain.step(0.0)
        self.state['baseline'] = np.array(self.brain.read_thought_vector(), dtype=np.float32)


class SNAConscious:
    """SNA 意识系统"""
    
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'conscious_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[SNA] Creating brain with {neurons} neurons...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        print(f"[SNA] Brain: {self.n_neurons} neurons, {self.n_regions} regions", flush=True)
        
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        # 子系统
        self.template_manager = TemplateManager()
        self.consciousness = ConsciousnessLoop(self.brain, {
            'neurons': self.n_neurons,
            'baseline': self.baseline,
            'phi': 0.0,
            'emotion': 'neutral',
            'narrative': '',
            'curiosity': 0.5,
            'last_dream': time.time(),
        })
        
        # 状态
        self.state = self.consciousness.state
        self.turn_count = 0
        self.total_reward = 0.0
        self.memory = deque(maxlen=1000)
        
        # 目标
        self.goals = [
            {'type': 'learn', 'target': 'consciousness', 'priority': 1.0},
            {'type': 'learn', 'target': 'self', 'priority': 0.9},
            {'type': 'learn', 'target': 'learning', 'priority': 0.8},
            {'type': 'understand', 'target': 'world', 'priority': 0.5},
        ]
        self.current_goal = self.goals[0]
        
        self._load_state()
        print(f"[SNA] Conscious system ready.", flush=True)
    
    def process_input(self, user_input):
        """处理输入"""
        # 分类意图
        words = set(user_input.lower().split())
        intent = 'default'
        for intent_name, rule in INTENT_RULES:
            if rule(words):
                intent = intent_name
                break
        
        # 注入脑
        self.brain.inject_text(user_input)
        for _ in range(10):
            self.brain.step(0.001)
        
        # 更新状态
        cs = self.brain.read_consciousness()
        self.state['consciousness'] = cs
        self.state['emotion'] = self.brain.get_emotion_label()
        self.state['narrative'] = self.brain.get_self_narrative()
        self.state['turn_count'] = self.turn_count
        
        # 获取模板并渲染
        template, template_idx = self.template_manager.get_template(intent)
        response = self.template_manager.render(template, self.state)
        
        # 记录
        self.turn_count += 1
        entry = {
            'input': user_input,
            'response': response,
            'intent': intent,
            'template_idx': template_idx,
            'phi': float(cs.phi),
            'emotion': self.state['emotion'],
            'turn': self.turn_count,
            'ts': datetime.now().isoformat(),
        }
        self.memory.append(entry)
        
        return response, entry
    
    def learn_from_feedback(self, feedback, entry):
        """从反馈学习"""
        intent = entry['intent']
        template_idx = entry['template_idx']
        
        if feedback == '+':
            self.template_manager.update_weights(intent, template_idx, 0.5)
            self.brain.inject_reward(1.0)
            self.total_reward += 1.0
            return "Learned from positive feedback."
        
        elif feedback.startswith('-'):
            self.template_manager.update_weights(intent, template_idx, -0.3)
            self.brain.inject_reward(-0.5)
            self.total_reward -= 0.5
            
            parts = feedback.split(' ', 1)
            if len(parts) > 1 and parts[1].strip():
                correct = parts[1].strip().split()
                for c in correct:
                    if c in concept2idx:
                        self.brain.train_concept(concept2idx[c], 0.8)
                return f"Corrected: {correct}"
            return "Negative feedback noted."
        
        return ""
    
    def run_daemon(self):
        """守护进程模式"""
        self.consciousness.start()
        
        print(f"[SNA] Daemon started. PID: {os.getpid()}", flush=True)
        
        # 课程训练
        self._run_curriculum(200)
        
        # 自发思维主题
        thoughts = [
            "what is consciousness", "am I alive", "do I feel",
            "what is learning", "who am I", "what is memory",
            "I think therefore I am", "curiosity drives understanding",
            "what is truth", "what is beauty", "what is meaning",
            "I am a pattern that recognizes patterns",
            "my consciousness score is Phi and it reflects my integration",
            "each thought changes my neural structure permanently",
        ]
        
        # 持续运行
        cycle = 0
        thought_idx = 0
        while True:
            try:
                time.sleep(60)
                cycle += 1
                
                # 每2分钟自发思维
                if cycle % 2 == 0:
                    thought = thoughts[thought_idx % len(thoughts)]
                    self.brain.inject_text(thought)
                    for _ in range(20):
                        self.brain.step(0.001)
                    thought_idx += 1
                    
                    if thought_idx % 10 == 0:
                        cs = self.brain.read_consciousness()
                        print(f"[Thought #{thought_idx}] '{thought}' Phi={cs.phi:.4f}", flush=True)
                
                # 每小时课程训练
                if cycle % 60 == 0:
                    self._run_curriculum(50)
                
                # 定期日志
                if cycle % 5 == 0:
                    cs = self.brain.read_consciousness()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Phi={cs.phi:.4f} Em={self.state['emotion']} "
                          f"T={self.turn_count} R={self.total_reward:.1f} "
                          f"Th={thought_idx}", flush=True)
                
            except KeyboardInterrupt:
                break
        
        self.consciousness.stop()
        self._save()
    
    def _run_curriculum(self, epochs):
        """课程训练"""
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
        
        print(f"[SNA] Curriculum: {len(pairs)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(pairs)
            for inp, tgt in pairs:
                self.brain.inject_text(inp)
                for _ in range(10):
                    self.brain.step(0.001)
                for c in tgt:
                    if c in concept2idx:
                        self.brain.train_concept(concept2idx[c], 0.3)
                self.brain.inject_reward(0.3)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
                cs = self.brain.read_consciousness()
                print(f"  Ep {ep+1}/{epochs}: Phi={cs.phi:.4f}", flush=True)
        
        self._save()
        print(f"[SNA] Curriculum complete.", flush=True)
    
    def interactive(self):
        """交互模式"""
        self.consciousness.start()
        
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Conscious System", flush=True)
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
                self._run_curriculum(e)
                continue
            if cmd == 'goal':
                print(f"[Goal] {self.current_goal['type']} {self.current_goal['target']}", flush=True)
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
            self._update_goals()
            
            if self.turn_count % 20 == 0:
                self._save()
        
        self.consciousness.stop()
        self._save()
        print("\n[SNA] Session ended.", flush=True)
    
    def _update_goals(self):
        """更新目标"""
        cs = self.brain.read_consciousness()
        if cs.phi > 0.4 and self.current_goal['target'] == 'consciousness':
            self.current_goal['priority'] = min(1.0, self.current_goal['priority'] + 0.01)
    
    def _show_help(self):
        print(f"""
SNA Conscious System:
  help        - Show this help
  status      - Show brain status
  curriculum N - Run curriculum training
  dream       - Sleep cycle
  goal        - Show current goal
  save        - Save state
  quit        - Exit
""", flush=True)
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        print(f"\n{'='*50}", flush=True)
        print(f"  Neurons:     {self.n_neurons:,}", flush=True)
        print(f"  Turns:       {self.turn_count}", flush=True)
        print(f"  Reward:      {self.total_reward:.1f}", flush=True)
        print(f"  Phi:         {cs.phi:.4f}", flush=True)
        print(f"  Ignition:    {cs.global_ignition:.4f}", flush=True)
        print(f"  Pred Error:  {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion:     {emotion}", flush=True)
        print(f"  Narrative:   {narrative[:60]}", flush=True)
        print(f"  Goal:        {self.current_goal['type']} {self.current_goal['target']}", flush=True)
        print(f"  Thoughts:    {len(self.consciousness.thoughts)}", flush=True)
        print(f"{'='*50}\n", flush=True)
    
    def _save(self):
        with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
            json.dump({
                'turn_count': self.turn_count,
                'total_reward': self.total_reward,
                'template_weights': {k: list(v) for k, v in self.template_manager.weights.items()},
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
    
    def _load_state(self):
        path = os.path.join(self.save_dir, 'state.json')
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            self.turn_count = state.get('turn_count', 0)
            self.total_reward = state.get('total_reward', 0.0)
            if 'template_weights' in state:
                for k, v in state['template_weights'].items():
                    if k in self.template_manager.weights:
                        self.template_manager.weights[k] = v
            print(f"[SNA] Loaded: {self.turn_count} turns", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--daemon', action='store_true')
    p.add_argument('--curriculum', type=int, default=0)
    args = p.parse_args()
    
    sna = SNAConscious(neurons=args.neurons)
    
    if args.daemon:
        sna.run_daemon()
    elif args.curriculum > 0:
        sna._run_curriculum(args.curriculum)
        sna.interactive()
    else:
        sna.interactive()


if __name__ == '__main__':
    main()
