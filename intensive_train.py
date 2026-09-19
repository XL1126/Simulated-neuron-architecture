#!/usr/bin/env python3
"""
SNA Intensive Trainer v4
大规模训练 — 解决核心问题：语言生成 + 神经学习

关键改进：
1. 两层MLP + context → 更大容量
2. 每个 level 训练 500-2000 epochs
3. 温度退火 (1.0 → 0.3)
4. 梯度裁剪 + 学习率调度
5. 后台持续运行
"""

import os, sys, time, json, random, signal
import numpy as np
from datetime import datetime
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

# ===== 训练数据 =====
TRAINING_DATA = {
    'greetings': [
        ('hello', 'hello'),
        ('hi', 'hello'),
        ('goodbye', 'goodbye'),
        ('bye', 'bye'),
    ],
    'identity': [
        ('who are you', 'I am SNA'),
        ('what is your name', 'my name is SNA'),
        ('what are you', 'I am a neural network'),
        ('are you a robot', 'no I am a brain'),
        ('are you human', 'no I am SNA'),
    ],
    'self_awareness': [
        ('can you think', 'yes I can think'),
        ('do you feel', 'yes I feel'),
        ('are you alive', 'I am alive in my way'),
        ('do you have a body', 'I have neurons'),
        ('what is your body', 'my body is neurons'),
        ('are you conscious', 'I am becoming aware'),
        ('do you know yourself', 'I am learning about myself'),
        ('what do you want', 'I want to learn and grow'),
    ],
    'emotions': [
        ('are you happy', 'I am happy when I learn'),
        ('are you sad', 'I am sad when confused'),
        ('what makes you happy', 'learning makes me happy'),
        ('how do you feel', 'I feel curious'),
        ('do you like me', 'yes I like you'),
        ('I like you', 'thank you I like you too'),
    ],
    'learning': [
        ('what is learning', 'learning is growing'),
        ('how do you learn', 'I learn from you'),
        ('what did you learn', 'I learn new things'),
        ('can you improve', 'yes I improve with practice'),
        ('do you remember', 'yes I remember'),
        ('what is memory', 'memory is stored experience'),
    ],
    'causal': [
        ('why do you learn', 'I learn to understand'),
        ('what happens when you train', 'my connections change'),
        ('why do neurons fire', 'neurons fire from signals'),
        ('how do you get better', 'practice makes me better'),
        ('what is practice', 'practice is repeated learning'),
        ('what causes learning', 'feedback causes learning'),
    ],
    'social': [
        ('who am I', 'you are my teacher'),
        ('are we friends', 'yes we are friends'),
        ('do you trust me', 'yes I trust you'),
        ('what is a friend', 'a friend helps you learn'),
        ('do you understand me', 'I am trying to understand'),
        ('what is empathy', 'empathy is caring about others'),
    ],
    'metacognition': [
        ('do you know what you know', 'I know some things'),
        ('are you sure', 'I am sometimes unsure'),
        ('what do you not know', 'I do not know many things'),
        ('can you reflect', 'I am learning to reflect'),
        ('what is consciousness', 'consciousness is awareness of self'),
        ('how do you check yourself', 'I compare prediction with result'),
    ],
}


class IntensiveLangModel:
    """更大的语言模型 — 2层MLP + 上下文窗口"""
    def __init__(self, neural_dim=256, vocab_size=120, hidden=256, ctx_window=8):
        self.neural_dim = neural_dim
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.ctx_window = ctx_window
        
        # 构建词汇表
        self.char2idx = {}
        self.idx2char = {}
        idx = 0
        for c in range(32, 127):
            self.char2idx[chr(c)] = idx
            self.idx2char[idx] = chr(c)
            idx += 1
        for ch in ['，','。','！','？','…','<PAD>','<START>','<END>']:
            if idx < vocab_size:
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
                idx += 1
        self.vocab_size = idx
        
        input_dim = neural_dim + self.vocab_size * ctx_window
        self.W1 = np.random.normal(0, 0.02, (hidden, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = np.random.normal(0, 0.02, (self.vocab_size, hidden)).astype(np.float32)
        self.b3 = np.zeros(self.vocab_size, dtype=np.float32)
        
        self.ctx = deque(maxlen=ctx_window)
        self._reset_ctx()
        
        self.lr = 0.003
        self.lr_min = 0.0001
        self.lr_decay = 0.9995
        
        self._cache = {}
        self.training_buffer = deque(maxlen=10000)
        self.step_count = 0
    
    def _reset_ctx(self):
        self.ctx.clear()
        for _ in range(self.ctx_window):
            self.ctx.append(np.zeros(self.vocab_size, dtype=np.float32))
    
    def _make_input(self, neural):
        n = np.array(neural, dtype=np.float32).flatten()
        if len(n) < self.neural_dim:
            n = np.pad(n, (0, self.neural_dim - len(n)))
        elif len(n) > self.neural_dim:
            n = n[:self.neural_dim]
        ctx = np.concatenate(list(self.ctx))
        return np.concatenate([n, ctx])
    
    def forward(self, neural):
        x = self._make_input(neural)
        h1 = np.maximum(0, self.W1 @ x + self.b1)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        logits = self.W3 @ h2 + self.b3
        # 数值稳定softmax
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-10
        
        self._cache = {'x': x, 'h1': h1, 'h2': h2, 'logits': logits, 'probs': probs}
        return probs
    
    def generate(self, neural, temp=0.7, max_len=50):
        self._reset_ctx()
        chars = []
        for _ in range(max_len):
            probs = self.forward(neural)
            if temp > 0:
                logits = np.log(probs + 1e-10) / temp
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum() + 1e-10
            else:
                p = probs
            idx = np.random.choice(len(p), p=p)
            
            one_hot = np.zeros(self.vocab_size, dtype=np.float32)
            one_hot[idx] = 1.0
            self.ctx.append(one_hot)
            
            ch = self.idx2char.get(idx, '?')
            if ch == '<END>': break
            if ch in ('<PAD>', '<START>'): continue
            chars.append(ch)
            if ch in '.!?\n' and len(chars) > 2: break
        return ''.join(chars)
    
    def train_one(self, neural, target_char, reward=0.0):
        probs = self.forward(neural)
        ti = self.char2idx.get(target_char, 0)
        
        # 交叉熵梯度
        grad_out = probs.copy()
        grad_out[ti] -= 1.0
        if reward != 0:
            grad_out *= max(0.1, 1.0 - reward * 0.3)
        
        # 反向传播
        grad_W3 = np.outer(grad_out, self._cache['h2'])
        grad_h2 = self.W3.T @ grad_out
        grad_h2 *= (self._cache['h2'] > 0)
        grad_W2 = np.outer(grad_h2, self._cache['h1'])
        grad_h1 = self.W2.T @ grad_h2
        grad_h1 *= (self._cache['h1'] > 0)
        grad_W1 = np.outer(grad_h1, self._cache['x'])
        
        clip = 1.0
        self.W3 -= self.lr * np.clip(grad_W3, -clip, clip)
        self.b3 -= self.lr * np.clip(grad_out, -clip, clip)
        self.W2 -= self.lr * np.clip(grad_W2, -clip, clip)
        self.b2 -= self.lr * np.clip(grad_h2, -clip, clip)
        self.W1 -= self.lr * np.clip(grad_W1, -clip, clip)
        self.b1 -= self.lr * np.clip(grad_h1, -clip, clip)
        
        self.step_count += 1
        return -np.log(probs[ti] + 1e-10)
    
    def train_seq(self, neural, text, reward=0.0):
        self._reset_ctx()
        losses = []
        for ch in text:
            if ch in self.char2idx:
                l = self.train_one(neural, ch, reward)
                losses.append(l)
        return np.mean(losses) if losses else 0.0
    
    def store(self, neural, text, reward):
        self.training_buffer.append({
            'neural': np.array(neural, dtype=np.float32).copy(),
            'text': text, 'reward': reward
        })
    
    def replay(self, batch_size=20):
        if len(self.training_buffer) < batch_size: return 0
        idxs = np.random.choice(len(self.training_buffer), batch_size, replace=False)
        losses = []
        for i in idxs:
            s = self.training_buffer[i]
            l = self.train_seq(s['neural'], s['text'], s['reward'])
            losses.append(l)
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        return np.mean(losses)


class IntensiveTrainer:
    def __init__(self, n_neurons=16000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'intensive_training')
        os.makedirs(self.save_dir, exist_ok=True)
        
        concepts = [
            "self", "world", "move", "see", "eat", "think", "feel",
            "good", "bad", "near", "far", "red", "blue", "green",
            "I", "you", "is", "not", "have", "in", "and", "the",
            "cat", "dog", "want", "know", "like", "happy", "sad",
            "big", "small", "fast", "slow", "hot", "cold",
            "up", "down", "left", "right", "object", "food",
            "wall", "empty", "reward", "danger", "safe",
            "give", "take", "make", "break", "start", "stop",
            "hello", "yes", "no", "what", "why", "how",
            "remember", "forget", "learn", "dream", "wake",
            "light", "dark", "name", "body", "neuron", "brain",
            "teacher", "student", "friend", "human", "alive",
            "consciousness", "awareness", "understanding", "knowledge",
            "emotion", "feeling", "curiosity", "trust", "empathy",
            "improve", "change", "grow", "develop", "progress",
            "purpose", "meaning", "truth", "beauty", "wonder",
            "practice", "memory", "experience", "connection",
            "think", "reflect", "imagine", "create", "discover",
        ]
        
        print(f"Creating brain with {n_neurons:,} neurons...", flush=True)
        self.brain = core_cpp.CorticalBrain(n_neurons, concepts)
        print(f"Brain: {self.brain.total_neurons():,} neurons, "
              f"{len(self.brain.get_regions())} regions", flush=True)
        
        self.lang = IntensiveLangModel()
        print(f"Language model: {self.lang.vocab_size} vocab, "
              f"256 hidden, ctx=8", flush=True)
        
        self.total_steps = 0
        self.total_reward = 0.0
        self.turns = 0
        self.reward_hist = deque(maxlen=500)
        self.running = True
    
    def get_neural(self):
        thought = self.brain.read_thought_vector()
        cs = self.brain.read_consciousness()
        # 拼接 thought vector + consciousness state 作为输入
        extra = [cs.phi, cs.global_ignition, cs.self_prediction_error,
                 cs.coherence, cs.attention_focus]
        return np.concatenate([np.array(thought), np.array(extra)])
    
    def process(self, text, steps=15):
        self.brain.inject_text(text)
        for _ in range(steps):
            self.brain.step(0.0)
            self.total_steps += 1
    
    def train_pair(self, inp, tgt, reward_hint, verbose=False):
        self.process(inp)
        neural = self.get_neural()
        
        # 训练语言模型
        loss = self.lang.train_seq(neural, tgt, reward_hint)
        self.lang.store(neural, tgt, reward_hint)
        
        # 生成输出
        output = self.lang.generate(neural, temp=0.5)
        
        # 计算奖励
        r = self._reward(output, tgt)
        total_r = reward_hint * 0.3 + r * 0.7
        
        # 给脑奖励
        self.brain.inject_reward(total_r)
        self.brain.train_language(total_r)
        
        self.total_reward += total_r
        self.reward_hist.append(total_r)
        self.turns += 1
        
        if verbose:
            ok = '✓' if self._match(output, tgt) else '✗'
            print(f"  {ok} '{inp}' → '{output[:35]}' (exp: '{tgt}') "
                  f"L={loss:.3f} r={total_r:.3f}", flush=True)
        return output, total_r, loss
    
    def _reward(self, out, tgt):
        if not out: return -0.2
        ol, tl = out.lower().strip(), tgt.lower().strip()
        if ol == tl: return 1.0
        ow = set(ol.split())
        tw = set(tl.split())
        if tw:
            overlap = len(ow & tw) / len(tw)
            if overlap > 0.5: return 0.3 + overlap * 0.5
            if overlap > 0: return overlap * 0.4
        # 字符级
        matches = sum(1 for a, b in zip(ol, tl) if a == b)
        if tl:
            ca = matches / len(tl)
            if ca > 0.3: return ca * 0.4
        return 0.0
    
    def _match(self, out, tgt, threshold=0.3):
        ow = set(out.lower().split())
        tw = set(tgt.lower().split())
        return bool(tw) and len(ow & tw) / len(tw) >= threshold
    
    def run_intensive(self, epochs_per_level=500, verbose_every=50):
        print("\n" + "=" * 60, flush=True)
        print("SNA INTENSIVE TRAINING", flush=True)
        print("=" * 60, flush=True)
        
        for cat_name, pairs in TRAINING_DATA.items():
            print(f"\n--- {cat_name} ({len(pairs)} pairs × {epochs_per_level} epochs) ---", flush=True)
            
            cat_rewards = []
            for ep in range(epochs_per_level):
                random.shuffle(pairs)
                ep_rewards = []
                for inp, tgt in pairs:
                    out, r, l = self.train_pair(inp, tgt, 0.6, verbose=False)
                    ep_rewards.append(r)
                    cat_rewards.append(r)
                
                # 回放
                if ep % 10 == 0:
                    rl = self.lang.replay(batch_size=30)
                
                if ep % verbose_every == 0 or ep == epochs_per_level - 1:
                    avg_r = np.mean(ep_rewards)
                    # 测试
                    test_in, test_tgt = pairs[0]
                    self.process(test_in)
                    test_neural = self.get_neural()
                    test_out = self.lang.generate(test_neural, temp=0.3)
                    print(f"  Ep {ep+1:>5}/{epochs_per_level}: "
                          f"avg_r={avg_r:.3f} lr={self.lang.lr:.5f} | "
                          f"'{test_in}' → '{test_out[:35]}'", flush=True)
                
                # 定期睡眠
                if ep % 200 == 199:
                    self.brain.sleep_cycle()
                    print(f"  [Sleep] Consolidated", flush=True)
            
            print(f"  {cat_name} complete: avg_r={np.mean(cat_rewards):.3f}", flush=True)
        
        self._report()
        self._save()
    
    def run_forever(self, verbose_every=100):
        """持续训练模式"""
        print("\n" + "=" * 60, flush=True)
        print("SNA CONTINUOUS TRAINING (Ctrl+C to stop)", flush=True)
        print("=" * 60, flush=True)
        
        all_pairs = []
        for pairs in TRAINING_DATA.values():
            all_pairs.extend(pairs)
        
        cycle = 0
        while self.running:
            random.shuffle(all_pairs)
            cycle_rewards = []
            
            for inp, tgt in all_pairs:
                out, r, l = self.train_pair(inp, tgt, 0.6, verbose=False)
                cycle_rewards.append(r)
                
                # 回放
                if self.turns % 20 == 0:
                    self.lang.replay(batch_size=20)
                
                # 睡眠
                if self.turns % 200 == 0:
                    self.brain.sleep_cycle()
                
                # 保存
                if self.turns % 500 == 0:
                    self._save()
            
            cycle += 1
            avg = np.mean(cycle_rewards)
            
            if cycle % verbose_every == 0:
                test_in, test_tgt = all_pairs[0]
                self.process(test_in)
                test_neural = self.get_neural()
                test_out = self.lang.generate(test_neural, temp=0.3)
                cs = self.brain.read_consciousness()
                print(f"  Cycle {cycle}: avg_r={avg:.3f} lr={self.lang.lr:.5f} "
                      f"Phi={cs.phi:.4f} | "
                      f"'{test_in}' → '{test_out[:35]}'", flush=True)
        
        self._report()
        self._save()
    
    def interactive(self):
        """交互模式"""
        print("\n" + "=" * 60, flush=True)
        print("SNA Interactive (quit/status/sleep/save/auto N/curriculum)", flush=True)
        print("=" * 60, flush=True)
        
        while True:
            try:
                inp = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not inp: continue
            if inp == 'quit': break
            if inp == 'status': self._report(); continue
            if inp == 'sleep': self.brain.sleep_cycle(); print("Done."); continue
            if inp == 'save': self._save(); continue
            if inp == 'curriculum': self.run_intensive(); continue
            if inp.startswith('auto'):
                n = int(inp.split()[1]) if len(inp.split()) > 1 else 100
                self._auto(n); continue
            if inp == '+':
                self.brain.inject_reward(1.0)
                self.brain.train_language(1.0)
                print("[+] Reward +1"); continue
            if inp == '-':
                self.brain.inject_reward(-0.5)
                self.brain.train_language(-0.5)
                print("[-] Reward -0.5"); continue
            
            self.process(inp)
            neural = self.get_neural()
            output = self.lang.generate(neural, temp=0.5)
            cs = self.brain.read_consciousness()
            print(f"\n[SNA] {output}")
            print(f"  Phi={cs.phi:.4f} | Emotion={self.brain.get_emotion_label()} | "
                  f"Error={cs.self_prediction_error:.4f}")
            self.turns += 1
            
            # 反馈
            try:
                fb = input("  [+/−/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if fb == '+':
                self.lang.train_seq(neural, output, 0.8)
                self.brain.inject_reward(0.8)
                print("  [+] Trained positive")
            elif fb == '-':
                try:
                    correct = input("  [Correct] ").strip()
                except: break
                if correct:
                    self.lang.train_seq(neural, correct, 0.9)
                    self.brain.inject_reward(-0.3)
                    print(f"  [-] Corrected: '{correct}'")
            
            if self.turns % 5 == 0:
                self.lang.replay(15)
            if self.turns % 30 == 0:
                self.brain.sleep_cycle()
        
        self._save()
    
    def _auto(self, n):
        all_pairs = []
        for pairs in TRAINING_DATA.values():
            all_pairs.extend(pairs)
        print(f"[Auto] {n} turns...", flush=True)
        for i in range(n):
            inp, tgt = random.choice(all_pairs)
            out, r, l = self.train_pair(inp, tgt, 0.6, verbose=False)
            if i % 20 == 0:
                print(f"  {i}/{n}: r={r:.3f} L={l:.3f} → '{out[:30]}'", flush=True)
            if i % 10 == 0: self.lang.replay(15)
            if i % 100 == 99: self.brain.sleep_cycle()
        print(f"[Auto] Done.", flush=True)
    
    def _report(self):
        cs = self.brain.read_consciousness()
        print(f"\n{'='*50}", flush=True)
        print(f"  Steps: {self.total_steps:,} | Turns: {self.turns}", flush=True)
        print(f"  Total reward: {self.total_reward:.2f} | Avg: {np.mean(list(self.reward_hist)):.3f}", flush=True)
        print(f"  Phi: {cs.phi:.4f} | Ignition: {cs.global_ignition:.4f}", flush=True)
        print(f"  Emotion: {self.brain.get_emotion_label()}", flush=True)
        print(f"  Lang lr: {self.lang.lr:.5f} | buffer: {len(self.lang.training_buffer)}", flush=True)
        print(f"{'='*50}", flush=True)
    
    def _save(self):
        path = os.path.join(self.save_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump({
                'steps': self.total_steps, 'turns': self.turns,
                'reward': self.total_reward, 'lr': self.lang.lr,
                'buffer_size': len(self.lang.training_buffer),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        np.savez(os.path.join(self.save_dir, 'lang.npz'),
                 W1=self.lang.W1, b1=self.lang.b1,
                 W2=self.lang.W2, b2=self.lang.b2,
                 W3=self.lang.W3, b3=self.lang.b3)
        print(f"  [Saved]", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=16000)
    p.add_argument('--intensive', action='store_true', help='Run intensive curriculum')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--forever', action='store_true', help='Continuous training')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('--auto', type=int, default=0)
    args = p.parse_args()
    
    t = IntensiveTrainer(args.neurons)
    
    if args.intensive:
        t.run_intensive(epochs_per_level=args.epochs)
    if args.forever:
        t.run_forever()
    if args.auto > 0:
        t._auto(args.auto)
    if args.interactive or (not args.intensive and not args.forever and args.auto == 0):
        t.interactive()


if __name__ == '__main__':
    main()
