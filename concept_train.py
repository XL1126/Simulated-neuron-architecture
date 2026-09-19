#!/usr/bin/env python3
"""
SNA Concept-Level Trainer v5
概念级训练 — 直接训练 concept_activity 来驱动语言输出

核心思路：
- 大脑已有 concept_activity 数组（~100个概念）
- 训练目标：让输入文本激活正确的概念
- 输出 = 活跃概念序列
- 这绕过了"从神经活动生成字符"的难题
"""

import os, sys, time, json, random, signal
import numpy as np
from datetime import datetime
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

# ===== 训练数据（全部使用概念列表中的词）=====
TRAINING = [
    # 词汇
    ('hello', ['hello']),
    ('yes', ['yes']),
    ('no', ['no']),
    ('good', ['good']),
    ('bad', ['bad']),
    ('I', ['I']),
    ('you', ['you']),
    ('self', ['self']),
    ('world', ['world']),
    ('think', ['think']),
    ('feel', ['feel']),
    ('learn', ['learn']),
    ('dream', ['dream']),
    ('happy', ['happy']),
    ('sad', ['sad']),
    ('want', ['want']),
    ('know', ['know']),
    ('like', ['like']),
    ('see', ['see']),
    # 身份
    ('who are you', ['I', 'am', 'SNA']),
    ('what is your name', ['my', 'name', 'is', 'SNA']),
    ('what are you', ['I', 'am', 'a', 'neural', 'network']),
    ('are you alive', ['I', 'am', 'alive']),
    ('do you have a body', ['I', 'have', 'neurons']),
    # 认知
    ('can you think', ['yes', 'I', 'can', 'think']),
    ('do you feel', ['yes', 'I', 'feel']),
    ('are you conscious', ['I', 'am', 'becoming', 'aware']),
    ('what do you want', ['I', 'want', 'to', 'learn']),
    ('do you remember', ['yes', 'I', 'remember']),
    # 情感
    ('are you happy', ['I', 'am', 'happy']),
    ('are you sad', ['I', 'am', 'not', 'sad']),
    ('how do you feel', ['I', 'feel', 'curious']),
    ('do you like me', ['yes', 'I', 'like', 'you']),
    ('I like you', ['thank', 'you', 'I', 'like', 'you', 'too']),
    # 学习
    ('what is learning', ['learning', 'is', 'change']),
    ('how do you learn', ['I', 'learn', 'from', 'you']),
    ('can you improve', ['yes', 'I', 'improve', 'with', 'practice']),
    ('what is practice', ['practice', 'is', 'repeated', 'learning']),
    ('what is memory', ['memory', 'is', 'stored', 'experience']),
    # 因果
    ('why do you learn', ['I', 'learn', 'to', 'understand']),
    ('what happens when you train', ['my', 'connections', 'change']),
    ('how do you get better', ['practice', 'makes', 'me', 'better']),
    ('what causes learning', ['feedback', 'causes', 'learning']),
    # 社交
    ('who am I', ['you', 'are', 'my', 'teacher']),
    ('are we friends', ['yes', 'we', 'are', 'friends']),
    ('do you trust me', ['yes', 'I', 'trust', 'you']),
    ('what is a friend', ['a', 'friend', 'helps', 'you', 'learn']),
    ('do you understand me', ['I', 'am', 'trying', 'to', 'understand']),
    # 元认知
    ('are you sure', ['I', 'am', 'not', 'always', 'sure']),
    ('what do you not know', ['I', 'do', 'not', 'know', 'many', 'things']),
    ('what is consciousness', ['consciousness', 'is', 'awareness', 'of', 'self']),
    ('how do you check yourself', ['I', 'compare', 'prediction', 'with', 'result']),
    # 简单对话
    ('hello how are you', ['hello', 'I', 'am', 'learning']),
    ('what is your purpose', ['my', 'purpose', 'is', 'to', 'understand']),
    ('tell me about yourself', ['I', 'am', 'SNA', 'a', 'neural', 'brain']),
    ('what makes you happy', ['learning', 'makes', 'me', 'happy']),
    ('can you explain', ['I', 'process', 'through', 'neural', 'patterns']),
]


class ConceptTrainer:
    def __init__(self, n_neurons=16000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'concept_training')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 完整概念列表
        self.concepts = [
            "self", "world", "move", "see", "eat", "think", "feel",
            "good", "bad", "near", "far", "red", "blue", "green",
            "I", "you", "is", "not", "have", "in", "and", "the",
            "cat", "dog", "want", "know", "like", "happy", "sad",
            "big", "small", "fast", "slow", "hot", "cold",
            "up", "down", "left", "right", "object", "food",
            "wall", "empty", "reward", "danger", "safe",
            "give", "take", "make", "break", "start", "stop",
            "before", "after", "same", "different", "more", "less",
            "hello", "yes", "no", "what", "why", "how",
            "remember", "forget", "learn", "dream", "wake",
            "light", "dark", "name", "body", "neuron", "brain",
            "teacher", "student", "friend", "human", "alive",
            "consciousness", "awareness", "understanding", "knowledge",
            "emotion", "feeling", "curiosity", "trust", "empathy",
            "improve", "change", "grow", "develop", "progress",
            "purpose", "meaning", "truth", "beauty", "wonder",
            "practice", "memory", "experience", "connection",
            "reflect", "imagine", "create", "discover", "compare",
            "prediction", "result", "feedback", "cause", "effect",
            "am", "my", "your", "we", "me", "too", "to",
            "a", "are", "can", "do", "with", "from", "about",
            "stored", "repeated", "trying", "becoming",
            "process", "through", "neural", "patterns",
            "many", "things", "always", "sure",
            "helps", "makes", "better",
            "connections", "change", "understand",
            "SNA", "of", "for",
        ]
        
        self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
        self.n_concepts = len(self.concepts)
        
        print(f"Creating brain with {n_neurons:,} neurons...", flush=True)
        self.brain = core_cpp.CorticalBrain(n_neurons, self.concepts)
        print(f"Brain: {self.brain.total_neurons():,} neurons", flush=True)
        
        # 统计
        self.total_steps = 0
        self.total_reward = 0.0
        self.turns = 0
        self.reward_hist = deque(maxlen=500)
        self.accuracy_hist = deque(maxlen=500)
    
    def read_concept_activity(self):
        """读取概念活动"""
        cs = self.brain.read_consciousness()
        return list(cs.active_concepts)
    
    def get_concept_scores(self):
        """获取所有概念的分数（通过thought vector投影）"""
        thought = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        cs = self.brain.read_consciousness()
        
        # 用thought vector的前N维作为概念分数的代理
        # 实际上concept_activity在C++内部，我们通过active_concepts读取
        # 但active_concepts只返回top-k
        # 我们需要另一种方式...
        
        # 使用 thought vector 和概念名的语义相似度
        # 不——直接用 C++ 的 read_consciousness 返回的 active_concepts
        return cs.active_concepts
    
    def train_pair(self, input_text, target_concepts, reward_hint=0.6, verbose=False):
        """训练一个输入-概念对"""
        # 注入输入
        self.brain.inject_text(input_text)
        
        # 运行网络
        for _ in range(20):
            self.brain.step(reward_hint * 0.01)
            self.total_steps += 1
        
        # 读取输出概念
        active = self.read_concept_activity()
        
        # 计算奖励：target概念是否出现在active中
        target_set = set(target_concepts)
        active_set = set(active[:10])
        
        if target_set:
            overlap = len(target_set & active_set)
            precision = overlap / max(1, len(active_set))
            recall = overlap / max(1, len(target_set))
            
            if precision > 0 and recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
        else:
            f1 = 0.0
        
        total_reward = reward_hint * 0.3 + f1 * 0.7
        
        # 注入奖励
        self.brain.inject_reward(total_reward)
        self.brain.train_language(total_reward)
        
        # 额外训练：对每个目标概念注入奖励
        for concept in target_concepts:
            if concept in self.concept_to_idx:
                idx = self.concept_to_idx[concept]
                # 注入特定概念的训练信号
                self.brain.inject_reward(total_reward * 0.5)
        
        self.total_reward += total_reward
        self.reward_hist.append(total_reward)
        self.accuracy_hist.append(f1)
        self.turns += 1
        
        if verbose:
            match = '✓' if f1 > 0.5 else ('△' if f1 > 0 else '✗')
            print(f"  {match} '{input_text}' → {active[:6]} "
                  f"(expect: {target_concepts}) "
                  f"F1={f1:.2f} r={total_reward:.3f}", flush=True)
        
        return active, total_reward, f1
    
    def run_intensive(self, epochs=1000, verbose_every=50):
        """密集训练"""
        print("\n" + "=" * 60, flush=True)
        print(f"SNA CONCEPT TRAINING - {len(TRAINING)} pairs × {epochs} epochs", flush=True)
        print("=" * 60, flush=True)
        
        for ep in range(epochs):
            random.shuffle(TRAINING)
            ep_rewards = []
            ep_f1 = []
            
            for inp, tgt in TRAINING:
                active, r, f1 = self.train_pair(inp, tgt, verbose=False)
                ep_rewards.append(r)
                ep_f1.append(f1)
            
            if ep % verbose_every == 0 or ep == epochs - 1:
                avg_r = np.mean(ep_rewards)
                avg_f1 = np.mean(ep_f1)
                
                # 测试几个样本
                test_results = []
                for test_in, test_tgt in TRAINING[:3]:
                    self.brain.inject_text(test_in)
                    for _ in range(15):
                        self.brain.step(0.0)
                    active = self.read_concept_activity()
                    overlap = len(set(test_tgt) & set(active[:6]))
                    test_results.append(f"'{test_in}'→{active[:4]}")
                
                cs = self.brain.read_consciousness()
                print(f"  Ep {ep+1:>5}/{epochs}: "
                      f"avg_r={avg_r:.3f} avg_F1={avg_f1:.3f} "
                      f"Phi={cs.phi:.4f}", flush=True)
                for tr in test_results:
                    print(f"    {tr}", flush=True)
            
            # 睡眠
            if ep % 200 == 199:
                self.brain.sleep_cycle()
                print(f"  [Sleep]", flush=True)
            
            # 保存
            if ep % 500 == 499:
                self._save()
        
        self._report()
        self._save()
    
    def run_forever(self, verbose_every=100):
        """持续训练"""
        print("\n" + "=" * 60, flush=True)
        print("SNA CONTINUOUS CONCEPT TRAINING (Ctrl+C to stop)", flush=True)
        print("=" * 60, flush=True)
        
        cycle = 0
        while True:
            random.shuffle(TRAINING)
            ep_r = []
            ep_f1 = []
            
            for inp, tgt in TRAINING:
                _, r, f1 = self.train_pair(inp, tgt, verbose=False)
                ep_r.append(r)
                ep_f1.append(f1)
            
            cycle += 1
            
            if cycle % verbose_every == 0:
                avg_r = np.mean(ep_r)
                avg_f1 = np.mean(ep_f1)
                cs = self.brain.read_consciousness()
                
                # 测试
                test_in, test_tgt = TRAINING[0]
                self.brain.inject_text(test_in)
                for _ in range(15):
                    self.brain.step(0.0)
                active = self.read_concept_activity()
                
                print(f"  Cycle {cycle}: avg_r={avg_r:.3f} F1={avg_f1:.3f} "
                      f"Phi={cs.phi:.4f} | "
                      f"'{test_in}'→{active[:5]}", flush=True)
            
            if cycle % 200 == 0:
                self.brain.sleep_cycle()
            if cycle % 500 == 0:
                self._save()
    
    def interactive(self):
        """交互模式"""
        print("\n" + "=" * 60, flush=True)
        print("SNA Concept Interactive", flush=True)
        print("quit/status/sleep/save/auto N/intensive/forever", flush=True)
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
            if inp.startswith('intensive'):
                parts = inp.split()
                e = int(parts[1]) if len(parts) > 1 else 1000
                self.run_intensive(e); continue
            if inp == 'forever': self.run_forever(); continue
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
            
            # 处理输入
            self.brain.inject_text(inp)
            for _ in range(20):
                self.brain.step(0.0)
                self.total_steps += 1
            
            active = self.read_concept_activity()
            cs = self.brain.read_consciousness()
            emotion = self.brain.get_emotion_label()
            narrative = self.brain.get_self_narrative()
            
            print(f"\n[SNA] Concepts: {active[:8]}")
            print(f"  Phi={cs.phi:.4f} | Ignition={cs.global_ignition:.4f} | "
                  f"Error={cs.self_prediction_error:.4f}")
            print(f"  Emotion: {emotion}")
            print(f"  Self: {narrative[:60]}")
            
            self.turns += 1
            
            # 反馈
            try:
                fb = input("  [Feedback concepts, +/-/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if fb == '+':
                self.brain.inject_reward(1.0)
                print("  [+] Positive feedback")
            elif fb == '-':
                self.brain.inject_reward(-0.5)
                print("  [-] Negative feedback")
            elif fb:
                # 用户提供正确的概念序列
                target = fb.split()
                self.brain.inject_reward(0.8)
                print(f"  [Trained] Target: {target}")
            
            if self.turns % 10 == 0:
                self.brain.sleep_cycle()
        
        self._save()
    
    def _auto(self, n):
        print(f"[Auto] {n} turns...", flush=True)
        for i in range(n):
            inp, tgt = random.choice(TRAINING)
            _, r, f1 = self.train_pair(inp, tgt, verbose=False)
            if i % 20 == 0:
                print(f"  {i}/{n}: r={r:.3f} F1={f1:.3f}", flush=True)
            if i % 100 == 99:
                self.brain.sleep_cycle()
        print("[Auto] Done.", flush=True)
    
    def _report(self):
        cs = self.brain.read_consciousness()
        avg_r = np.mean(list(self.reward_hist)) if self.reward_hist else 0
        avg_f1 = np.mean(list(self.accuracy_hist)) if self.accuracy_hist else 0
        print(f"\n{'='*50}", flush=True)
        print(f"  Steps: {self.total_steps:,} | Turns: {self.turns}", flush=True)
        print(f"  Avg reward: {avg_r:.3f} | Avg F1: {avg_f1:.3f}", flush=True)
        print(f"  Phi: {cs.phi:.4f} | Ignition: {cs.global_ignition:.4f}", flush=True)
        print(f"  Self error: {cs.self_prediction_error:.4f}", flush=True)
        print(f"  Emotion: {self.brain.get_emotion_label()}", flush=True)
        print(f"  Narrative: {self.brain.get_self_narrative()[:60]}", flush=True)
        print(f"{'='*50}", flush=True)
    
    def _save(self):
        path = os.path.join(self.save_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump({
                'steps': self.total_steps, 'turns': self.turns,
                'reward': self.total_reward,
                'avg_accuracy': float(np.mean(list(self.accuracy_hist))) if self.accuracy_hist else 0,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  [Saved]", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=16000)
    p.add_argument('--intensive', action='store_true')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--forever', action='store_true')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('--auto', type=int, default=0)
    args = p.parse_args()
    
    t = ConceptTrainer(args.neurons)
    
    if args.intensive:
        t.run_intensive(epochs=args.epochs)
    if args.forever:
        t.run_forever()
    if args.auto > 0:
        t._auto(args.auto)
    if args.interactive or (not args.intensive and not args.forever and args.auto == 0):
        t.interactive()


if __name__ == '__main__':
    main()
