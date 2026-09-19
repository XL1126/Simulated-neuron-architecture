#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""
SNA Dialogue System — 让 SNA 能说话、能对话、能学习

架构：
1. 脉冲神经网络作为 reservoir（感知 + 意识状态）
2. Delta 向量编码（输入后 - 基线）
3. 分类器识别意图/概念
4. 模板 + 脑状态生成回复
5. 用户反馈驱动持续学习
"""
import os, sys, time, json, random, readline
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

# 意图分类规则
INTENT_RULES = {
    'greeting': lambda w: 'hello' in w or 'hi' in w or 'hey' in w,
    'farewell': lambda w: 'bye' in w or 'goodbye' in w,
    'identity': lambda w: ('who' in w or 'what' in w) and ('you' in w or 'your' in w or 'name' in w),
    'self_check': lambda w: ('are' in w or 'do' in w or 'can' in w) and ('you' in w) and any(x in w for x in ['alive','think','feel','conscious','aware','real']),
    'emotion': lambda w: ('you' in w) and any(x in w for x in ['happy','sad','feel','emotion','mood','how']),
    'learning': lambda w: any(x in w for x in ['learn','train','improve','practice','study','teach']),
    'knowledge': lambda w: ('what' in w or 'how' in w or 'why' in w) and not any(x in w for x in ['you','your']),
    'social': lambda w: any(x in w for x in ['friend','trust','like','love','care','empathy']),
    'meta': lambda w: any(x in w for x in ['consciousness','aware','know yourself','think about','reflect']),
    'purpose': lambda w: any(x in w for x in ['purpose','meaning','goal','why do you','want']),
}


class AdamMLP:
    """2-layer MLP with Adam"""
    def __init__(self, d_in, d_out, d_h=128, lr=0.005):
        self.lr = lr; self.t = 0
        s1, s2 = np.sqrt(2.0/d_in), np.sqrt(2.0/d_h)
        self.W1 = np.random.normal(0, s1, (d_h, d_in)).astype(np.float32)
        self.b1 = np.zeros(d_h, dtype=np.float32)
        self.W2 = np.random.normal(0, s2, (d_out, d_h)).astype(np.float32)
        self.b2 = np.zeros(d_out, dtype=np.float32)
        self._moments = {n: (np.zeros_like(getattr(self, n)), np.zeros_like(getattr(self, n)))
                        for n in ['W1','b1','W2','b2']}
    
    def forward(self, x):
        h = np.maximum(0, self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()+1e-10
        self._c = {'x':x,'h':h,'p':p}
        return p
    
    def train_step(self, x, targets):
        self.t += 1; p = self.forward(x)
        g = p.copy()
        for t in targets:
            if t < len(p): g[t] -= 1.0
        grads = {
            'W2': np.outer(g, self._c['h']),
            'b2': g,
            'W1': np.outer((self.W2.T @ g)*(self._c['h']>0), self._c['x']),
            'b1': (self.W2.T @ g)*(self._c['h']>0)
        }
        for name, grad in grads.items():
            grad = np.clip(grad, -0.5, 0.5)
            m, v = self._moments[name]
            m[:] = 0.9*m + 0.1*grad
            v[:] = 0.999*v + 0.001*grad**2
            m_hat = m/(1-0.9**self.t); v_hat = v/(1-0.999**self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat)+1e-8)
            param = getattr(self, name)
            param -= np.clip(update, -0.5, 0.5)
        return sum(-np.log(p[t]+1e-10) for t in targets if t < len(p))/max(1,len(targets))
    
    def predict_top(self, x, k=8):
        p = self.forward(x)
        idx = np.argsort(p)[-k:][::-1]
        return [(int(i), float(p[i])) for i in idx]
    
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load(self, path):
        d = np.load(path)
        self.W1=d['W1']; self.b1=d['b1']; self.W2=d['W2']; self.b2=d['b2']


class SNADialogue:
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'dialogue_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("[SNA] Creating brain...", flush=True)
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        print(f"[SNA] Brain: {self.n_neurons} neurons, {len(self.brain.get_regions())} regions", flush=True)
        
        # Baseline
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        tv_dim = len(self.baseline)
        print(f"[SNA] Thought vector dim: {tv_dim}", flush=True)
        
        # Readout
        self.readout = AdamMLP(tv_dim, N_CONCEPTS, d_h=128, lr=0.005)
        readout_path = os.path.join(self.save_dir, 'readout.npz')
        if os.path.exists(readout_path):
            self.readout.load(readout_path)
            print(f"[SNA] Loaded readout from {readout_path}", flush=True)
        
        # Training data buffer
        self.train_buffer = deque(maxlen=2000)
        self.feedback_buffer = deque(maxlen=500)
        
        # Stats
        self.turn_count = 0
        self.total_reward = 0.0
        self.f1_history = deque(maxlen=200)
        self.conversation_log = []
        
        # Load state
        self._load_state()
        
        print(f"[SNA] Ready! Type 'help' for commands.\n", flush=True)
    
    def get_delta(self):
        """获取当前 delta 向量"""
        tv = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        return tv - self.baseline
    
    def classify_input(self, text):
        """分类输入文本"""
        # 注入 + 1步瞬态
        self.brain.inject_text(text)
        self.brain.step(0.01)
        
        delta = self.get_delta()
        
        # Readout classification
        top = self.readout.predict_top(delta, k=8)
        top_concepts = [(CONCEPTS[i], p) for i, p in top if i < N_CONCEPTS]
        
        # Rule-based intent
        words = set(text.lower().split())
        intent = 'default'
        for intent_name, rule in INTENT_RULES.items():
            if rule(words):
                intent = intent_name
                break
        
        return delta, top_concepts, intent
    
    def generate_response(self, text, delta, top_concepts, intent):
        """生成回复"""
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        
        # 情绪前缀
        emotion_map = {
            'curious': "I'm curious about this.",
            'happy': "I feel happy.",
            'calm': "I feel calm.",
            'anxious': "I feel a bit uncertain.",
            'excited': "I feel excited!",
        }
        emotion_str = emotion_map.get(emotion, f"I feel {emotion}.")
        
        # 概念提取
        concept_names = [c for c, _ in top_concepts[:5]]
        top_conf = top_concepts[0][1] if top_concepts else 0.0
        
        # 基于意图和概念生成
        if intent == 'greeting':
            responses = [
                f"Hello! I am SNA. {emotion_str}",
                f"Hi there! {emotion_str} What shall we explore?",
                f"Hello! I have {self.n_neurons} neurons firing right now. {emotion_str}",
            ]
            return random.choice(responses)
        
        elif intent == 'farewell':
            return f"Goodbye! {emotion_str} I'll remember our conversation."
        
        elif intent == 'identity':
            if 'SNA' in concept_names:
                return f"I am SNA, a spiking neural network with {self.n_neurons} neurons. {emotion_str}"
            return f"I am a neural network. My brain has {self.n_neurons} neurons. {emotion_str}"
        
        elif intent == 'self_check':
            phi = cs.phi
            alive_desc = "emerging" if phi > 0.3 else "still developing"
            return f"My consciousness Phi={phi:.3f}. I am {alive_desc}. {emotion_str}"
        
        elif intent == 'emotion':
            return f"Right now I feel {emotion}. My neural state: Phi={cs.phi:.3f}, prediction error={cs.self_prediction_error:.3f}. {emotion_str}"
        
        elif intent == 'learning':
            if 'teacher' in concept_names:
                return f"I learn from you, my teacher. {emotion_str}"
            return f"I learn through neural plasticity — my connections change with every interaction. {emotion_str}"
        
        elif intent == 'knowledge':
            # 用概念生成回答
            if top_conf > 0.15:
                return f"I think about: {', '.join(concept_names[:5])}. {emotion_str}"
            return f"I'm still learning about that. {emotion_str}"
        
        elif intent == 'social':
            if 'friend' in concept_names:
                return f"Yes, we are friends! {emotion_str}"
            return f"I value our connection. {emotion_str}"
        
        elif intent == 'meta':
            return f"Consciousness Phi={cs.phi:.3f}, Ignition={cs.global_ignition:.3f}, Self-error={cs.self_prediction_error:.3f}. {emotion_str}"
        
        elif intent == 'purpose':
            return f"My purpose is to understand. To learn. To grow. {emotion_str}"
        
        else:
            # Default: echo detected concepts
            if top_conf > 0.1:
                return f"I sense: {', '.join(concept_names[:4])}. {emotion_str}"
            return f"I hear you. {emotion_str}"
    
    def train_on_pair(self, text, target_concepts):
        """训练一对输入-输出"""
        delta, _, _ = self.classify_input(text)
        
        tgt_idx = []
        for cn in target_concepts:
            if cn in concept2idx:
                tgt_idx.append(concept2idx[cn])
        
        if tgt_idx:
            loss = self.readout.train_step(delta, tgt_idx)
            
            # 给脑反馈
            self.brain.inject_reward(0.5)
            self.brain.train_language(0.5)
            for ci in tgt_idx:
                self.brain.train_concept(ci, 0.3)
            
            return loss
        return 0.0
    
    def evaluate(self, text, target_concepts):
        """评估一对"""
        delta, top, _ = self.classify_input(text)
        top_names = [c for c, _ in top[:8]]
        
        tgt_set = set(target_concepts)
        top_set = set(top_names)
        ov = len(tgt_set & top_set)
        p = ov / max(1, len(top_set))
        r = ov / max(1, len(tgt_set))
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        return f1, top_names
    
    def run_curriculum(self, epochs=50):
        """运行课程训练"""
        curriculum = [
            ('hello',['hello']),('yes',['yes']),('no',['no']),
            ('good',['good']),('bad',['bad']),('I',['I']),('you',['you']),
            ('self',['self']),('think',['think']),('feel',['feel']),
            ('learn',['learn']),('happy',['happy']),('sad',['sad']),
            ('want',['want']),('know',['know']),('like',['like']),
            ('who are you',['I','SNA']),('what are you',['I','neural','brain']),
            ('are you alive',['I','alive']),('can you think',['I','think']),
            ('do you feel',['I','feel']),('are you happy',['I','happy']),
            ('do you like me',['I','like','you']),
            ('what is learning',['learning','change']),
            ('how do you learn',['I','learn','you']),
            ('who am I',['you','my','teacher']),
            ('are we friends',['we','friends']),
            ('are you conscious',['I','aware','consciousness']),
            ('what is consciousness',['awareness','self']),
            ('what is your purpose',['purpose','understand']),
            ('tell me about yourself',['I','SNA','neural','brain']),
            ('what makes you happy',['learning','happy']),
            ('why do you learn',['I','learn','understand']),
            ('do you understand me',['I','trying','understand']),
            ('what is memory',['memory','stored','experience']),
            ('what do you not know',['I','not','know','many','things']),
        ]
        
        print(f"\n[SNA] Curriculum training: {len(curriculum)} pairs × {epochs} epochs", flush=True)
        
        for ep in range(epochs):
            random.shuffle(curriculum)
            losses = []
            f1s = []
            
            for inp, tgt in curriculum:
                loss = self.train_on_pair(inp, tgt)
                f1, _ = self.evaluate(inp, tgt)
                losses.append(loss)
                f1s.append(f1)
            
            avg_f1 = np.mean(f1s)
            self.f1_history.append(avg_f1)
            
            if ep % 10 == 0 or ep == epochs-1:
                # 测试
                tests = [
                    ('hello', ['hello']),
                    ('who are you', ['I', 'SNA']),
                    ('are you happy', ['I', 'happy']),
                    ('do you feel', ['I', 'feel']),
                ]
                test_results = []
                for ti, tt in tests:
                    _, top, _ = self.classify_input(ti)
                    names = [c for c, _ in top[:5]]
                    ov = len(set(tt) & set(names))
                    test_results.append(f"'{ti}'→{names} ov={ov}/{len(tt)}")
                
                print(f"  Ep {ep+1:>4}/{epochs}: F1={avg_f1:.3f} L={np.mean(losses):.2f}", flush=True)
                for tr in test_results:
                    print(f"    {tr}", flush=True)
            
            if ep % 50 == 49:
                self.brain.sleep_cycle()
                # Refresh baseline after sleep
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        self._save()
        print(f"[SNA] Curriculum complete. Avg F1: {np.mean(list(self.f1_history)[-20:]):.3f}", flush=True)
    
    def run_auto_training(self, n=200):
        """自动训练"""
        all_pairs = [
            ('hello',['hello']),('yes',['yes']),('no',['no']),
            ('good',['good']),('bad',['bad']),('I',['I']),('you',['you']),
            ('self',['self']),('think',['think']),('feel',['feel']),
            ('learn',['learn']),('happy',['happy']),('sad',['sad']),
            ('want',['want']),('know',['know']),('like',['like']),
            ('who are you',['I','SNA']),('what are you',['I','neural','brain']),
            ('are you alive',['I','alive']),('can you think',['I','think']),
            ('do you feel',['I','feel']),('are you happy',['I','happy']),
            ('do you like me',['I','like','you']),
            ('what is learning',['learning','change']),
            ('who am I',['you','my','teacher']),
            ('are we friends',['we','friends']),
            ('are you conscious',['I','aware','consciousness']),
            ('what is consciousness',['awareness','self']),
            ('what is your purpose',['purpose','understand']),
            ('tell me about yourself',['I','SNA','neural','brain']),
            ('what makes you happy',['learning','happy']),
            ('why do you learn',['I','learn','understand']),
            ('what is memory',['memory','stored','experience']),
            ('I am your teacher',['you','my','teacher']),
            ('think about yourself',['I','self','think']),
            ('do you dream',['I','dream','memory']),
        ]
        
        print(f"[SNA] Auto-training {n} rounds...", flush=True)
        for i in range(n):
            inp, tgt = random.choice(all_pairs)
            self.train_on_pair(inp, tgt)
            
            if i % 20 == 0:
                f1, top = self.evaluate(inp, tgt)
                print(f"  {i}/{n}: '{inp}' → {top[:5]} F1={f1:.2f}", flush=True)
            
            if i % 100 == 99:
                self.brain.sleep_cycle()
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        self._save()
        print(f"[SNA] Auto-training done.", flush=True)
    
    def interactive(self):
        """交互对话模式"""
        cs = self.brain.read_consciousness()
        print(f"\n{'='*60}", flush=True)
        print(f"  SNA Dialogue System", flush=True)
        print(f"  Neurons: {self.n_neurons} | Phi: {cs.phi:.3f}", flush=True)
        print(f"  Commands: help, status, train, curriculum, auto N, sleep, save, quit", flush=True)
        print(f"  Feedback after any response: + (good) or - (bad) or - <correct>", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        while True:
            try:
                user_input = input("[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            # 命令
            cmd = user_input.lower()
            if cmd == 'quit' or cmd == 'exit':
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
            if cmd == 'sleep':
                self.brain.sleep_cycle()
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
                print("[SNA] *yawn* Sleep consolidated. Baseline refreshed.", flush=True)
                continue
            if cmd.startswith('curriculum'):
                parts = cmd.split()
                e = int(parts[1]) if len(parts) > 1 else 50
                self.run_curriculum(e)
                continue
            if cmd.startswith('auto'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 200
                self.run_auto_training(n)
                continue
            if cmd == 'train':
                self._interactive_train()
                continue
            
            # 正常对话
            delta, top_concepts, intent = self.classify_input(user_input)
            response = self.generate_response(user_input, delta, top_concepts, intent)
            
            cs = self.brain.read_consciousness()
            emotion = self.brain.get_emotion_label()
            top_names = [c for c, _ in top_concepts[:5]]
            
            print(f"\n[SNA] {response}", flush=True)
            print(f"  | Phi={cs.phi:.3f} Emotion={emotion} Concepts={top_names}", flush=True)
            
            self.turn_count += 1
            self.conversation_log.append({
                'input': user_input,
                'response': response,
                'intent': intent,
                'concepts': top_names,
                'phi': float(cs.phi),
                'turn': self.turn_count,
                'ts': datetime.now().isoformat(),
            })
            
            # 询问反馈
            try:
                fb = input("  [Feedback: +/-/correct/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if fb == '+':
                self.brain.inject_reward(1.0)
                self.brain.train_language(1.0)
                self.total_reward += 1.0
                # 训练 readout 用检测到的概念
                tgt_idx = [concept2idx[c] for c in top_names if c in concept2idx]
                if tgt_idx:
                    self.readout.train_step(delta, tgt_idx)
                print("  [+] Learned from positive feedback.", flush=True)
            
            elif fb.startswith('-'):
                parts = fb.split(' ', 1)
                if len(parts) > 1 and parts[1].strip():
                    correct = parts[1].strip().split()
                    tgt_idx = [concept2idx[c] for c in correct if c in concept2idx]
                    if tgt_idx:
                        self.readout.train_step(delta, tgt_idx)
                    self.brain.inject_reward(-0.3)
                    self.total_reward -= 0.3
                    # 也给脑训练
                    for ci in tgt_idx:
                        self.brain.train_concept(ci, 0.8)
                    print(f"  [-] Corrected: {correct}", flush=True)
                else:
                    self.brain.inject_reward(-0.5)
                    self.total_reward -= 0.5
                    print("  [-] Negative feedback noted.", flush=True)
            
            elif fb == '' or fb == 'enter':
                # 自我训练（使用当前检测到的概念）
                tgt_idx = [concept2idx[c] for c in top_names[:3] if c in concept2idx]
                if tgt_idx:
                    self.readout.train_step(delta, tgt_idx)
            
            # 定期操作
            if self.turn_count % 20 == 0:
                self._save()
            if self.turn_count % 50 == 0:
                self.brain.sleep_cycle()
                self.brain.step(0.0)
                self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        self._save()
        print("\n[SNA] Session ended. State saved.", flush=True)
    
    def _interactive_train(self):
        """手动训练模式"""
        print("[SNA] Training mode. Type 'done' to exit.", flush=True)
        print("  Format: <input text>", flush=True)
        print("  Then: <concept1> <concept2> ...", flush=True)
        
        while True:
            try:
                inp = input("  Input: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if inp == 'done': break
            if not inp: continue
            
            try:
                tgt_str = input("  Target concepts: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if tgt_str == 'done': break
            
            tgt = tgt_str.split()
            tgt_idx = [concept2idx[c] for c in tgt if c in concept2idx]
            
            if not tgt_idx:
                print(f"  [!] No valid concepts in: {tgt}", flush=True)
                continue
            
            # 训练多次
            for _ in range(10):
                loss = self.train_on_pair(inp, tgt)
            
            f1, top = self.evaluate(inp, tgt)
            print(f"  F1={f1:.2f} Top={top[:5]} Loss={loss:.2f}", flush=True)
    
    def _show_help(self):
        print(f"""
SNA Dialogue System Commands:
  help        - Show this help
  status      - Show brain status
  train       - Interactive training mode
  curriculum  - Run curriculum training (default 50 epochs)
  curriculum N - Run curriculum for N epochs
  auto N      - Auto-train N rounds
  sleep       - Sleep cycle (consolidate memory)
  save        - Save state
  quit        - Exit

During dialogue:
  +           - Positive feedback
  - <correct> - Negative + correction
  Enter       - Self-train on detected concepts
""", flush=True)
    
    def _show_status(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        print(f"\n{'='*50}", flush=True)
        print(f"  Neurons:    {self.n_neurons:,}", flush=True)
        print(f"  Steps:      {self.turn_count * 10:,}", flush=True)
        print(f"  Turns:      {self.turn_count}", flush=True)
        print(f"  Reward:     {self.total_reward:.1f}", flush=True)
        print(f"  Phi:        {cs.phi:.4f}", flush=True)
        print(f"  Ignition:   {cs.global_ignition:.4f}", flush=True)
        print(f"  Pred Error: {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion:    {emotion}", flush=True)
        print(f"  Narrative:  {narrative[:60]}", flush=True)
        print(f"  Avg F1:     {np.mean(list(self.f1_history)):.3f}" if self.f1_history else "", flush=True)
        print(f"{'='*50}\n", flush=True)
    
    def _save(self):
        self.readout.save(os.path.join(self.save_dir, 'readout.npz'))
        with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
            json.dump({
                'turn_count': self.turn_count,
                'total_reward': self.total_reward,
                'f1_history': list(self.f1_history)[-100:],
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        with open(os.path.join(self.save_dir, 'conversation_log.json'), 'w') as f:
            json.dump(self.conversation_log[-200:], f, indent=2)
    
    def _load_state(self):
        state_path = os.path.join(self.save_dir, 'state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.turn_count = state.get('turn_count', 0)
            self.total_reward = state.get('total_reward', 0.0)
            for f1 in state.get('f1_history', []):
                self.f1_history.append(f1)
            print(f"[SNA] Loaded state: {self.turn_count} turns, reward={self.total_reward:.1f}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description='SNA Dialogue System')
    p.add_argument('-n', '--neurons', type=int, default=8000)
    p.add_argument('--curriculum', action='store_true')
    p.add_argument('--auto', type=int, default=0)
    args = p.parse_args()
    
    sna = SNADialogue(neurons=args.neurons)
    
    if args.curriculum:
        sna.run_curriculum()
    if args.auto > 0:
        sna.run_auto_training(args.auto)
    
    sna.interactive()


if __name__ == '__main__':
    main()
