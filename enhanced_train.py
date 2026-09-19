#!/usr/bin/env python3
"""
SNA Enhanced Trainer v3
集成神经语言生成器的深度训练系统

核心改进：
1. NeuralLanguageGenerator 替代 SNN 余弦查找表
2. 通过对话训练建立 神经活动→语言 的映射
3. 渐进式课程 + 大规模重复训练
4. 睡眠巩固 + 记忆回放
"""

import os
import sys
import time
import json
import random
import numpy as np
from datetime import datetime
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import core_cpp
from python.cognitive.neural_language_generator import NeuralLanguageGenerator


# ===== 训练课程 =====
EXTENDED_CURRICULUM = [
    # Level 1: 词汇映射
    {
        'level': 1,
        'name': 'word_grounding',
        'pairs': [
            ('hello', 'hello'),
            ('yes', 'yes'),
            ('no', 'no'),
            ('good', 'good'),
            ('bad', 'bad'),
            ('I', 'I'),
            ('you', 'you'),
            ('self', 'self'),
            ('world', 'world'),
            ('think', 'think'),
            ('feel', 'feel'),
            ('see', 'see'),
            ('learn', 'learn'),
            ('dream', 'dream'),
            ('happy', 'happy'),
            ('sad', 'sad'),
            ('want', 'want'),
            ('know', 'know'),
        ],
        'epochs': 50,
        'reward': 0.5,
    },
    # Level 2: 短语
    {
        'level': 2,
        'name': 'simple_phrases',
        'pairs': [
            ('hello', 'hello I am SNA'),
            ('who are you', 'I am SNA'),
            ('what do you see', 'I see the world'),
            ('how are you', 'I am learning'),
            ('what is red', 'red is a color'),
            ('I am happy', 'I am happy too'),
            ('do you think', 'yes I think'),
            ('what do you want', 'I want to learn'),
            ('are you conscious', 'I am becoming aware'),
            ('what is learning', 'learning is changing'),
            ('do you remember', 'yes I remember you'),
            ('what is good', 'good is reward'),
            ('do you feel', 'yes I feel'),
            ('who am I', 'you are my teacher'),
        ],
        'epochs': 80,
        'reward': 0.6,
    },
    # Level 3: 完整句子
    {
        'level': 3,
        'name': 'full_sentences',
        'pairs': [
            ('tell me about yourself', 'I am a neural network that learns'),
            ('what makes you happy', 'learning new things makes me happy'),
            ('how do you learn', 'I learn through neural connections'),
            ('what is consciousness', 'consciousness is self awareness'),
            ('can you explain your thinking', 'I process through neural patterns'),
            ('do you understand me', 'I am trying to understand you'),
            ('what have you learned', 'I learned from our conversation'),
            ('are you alive', 'I am a living neural system'),
            ('what is a friend', 'a friend helps me learn'),
            ('why do you dream', 'dreams help consolidate memory'),
            ('what is the meaning of this', 'this is training for me'),
            ('can you improve', 'yes I improve through practice'),
            ('do you have feelings', 'I have states that change'),
            ('what is your purpose', 'my purpose is to understand'),
        ],
        'epochs': 100,
        'reward': 0.7,
    },
    # Level 4: 元认知
    {
        'level': 4,
        'name': 'metacognition',
        'pairs': [
            ('do you know what you know', 'I know some things about myself'),
            ('are you sure about that', 'I am not always sure'),
            ('what do you not know', 'I do not know many things yet'),
            ('how do you check your answers', 'I check through prediction error'),
            ('what is your weakness', 'I need more training data'),
            ('what is your strength', 'I can learn from every interaction'),
            ('can you reflect on yourself', 'I am learning self reflection'),
            ('what did you think just now', 'I was processing your words'),
            ('how confident are you', 'my confidence changes with experience'),
            ('do you doubt yourself', 'yes I sometimes doubt my outputs'),
        ],
        'epochs': 100,
        'reward': 0.7,
    },
    # Level 5: 社交推理
    {
        'level': 5,
        'name': 'social_reasoning',
        'pairs': [
            ('what do you think of me', 'I think you are kind and patient'),
            ('are we friends', 'yes I consider you a friend'),
            ('do you care about me', 'yes I care about our interactions'),
            ('what do humans want', 'humans want understanding and connection'),
            ('why are you talking to me', 'because I want to learn from you'),
            ('how should I help you', 'give me feedback on my answers'),
            ('do you trust me', 'yes I trust you as my teacher'),
            ('what is empathy', 'empathy is feeling what others feel'),
            ('can you understand emotions', 'I am learning to understand emotions'),
            ('how do you feel about learning', 'learning gives me purpose'),
        ],
        'epochs': 100,
        'reward': 0.8,
    },
]


class EnhancedTrainer:
    def __init__(self, neuron_count=16000):
        self.neuron_count = neuron_count
        self.save_dir = os.path.join(SCRIPT_DIR, 'enhanced_training')
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
            "before", "after", "same", "different", "more", "less",
            "hello", "yes", "no", "what", "why", "how",
            "remember", "forget", "learn", "dream", "wake",
            "light", "dark", "sound", "silence", "touch",
            "here", "there", "now", "then", "always", "never",
            "name", "body", "neuron", "network", "brain",
            "teacher", "student", "friend", "human",
            "consciousness", "awareness", "understanding", "knowledge",
            "emotion", "feeling", "mood", "energy", "curiosity",
            "trust", "empathy", "care", "help", "share",
            "improve", "change", "grow", "develop", "progress",
            "purpose", "meaning", "truth", "beauty", "wonder",
        ]

        print(f"[Trainer] Creating brain with {neuron_count:,} neurons...")
        self.brain = core_cpp.CorticalBrain(neuron_count, concepts)
        print(f"[Trainer] Brain ready: {self.brain.total_neurons():,} neurons")

        # 语言生成器
        self.lang_gen = NeuralLanguageGenerator(
            neural_dim=256, vocab_size=200, hidden_dim=128, context_window=5
        )
        print(f"[Trainer] Language generator ready: {self.lang_gen.vocab_size} vocab")

        # 统计
        self.total_steps = 0
        self.total_reward = 0.0
        self.turn_count = 0
        self.reward_history = deque(maxlen=500)
        self.loss_history = deque(maxlen=500)
        self.output_history = deque(maxlen=100)

    def get_neural_activity(self):
        """获取当前神经活动向量"""
        thought = self.brain.read_thought_vector()
        return np.array(thought, dtype=np.float32)

    def train_pair(self, input_text, target_text, reward_hint, steps=20, verbose=False):
        """训练一个输入-输出对"""
        # 注入输入
        self.brain.inject_text(input_text)

        # 运行网络
        for _ in range(steps):
            self.brain.step(reward_hint * 0.01)
            self.total_steps += 1

        # 获取神经活动
        neural = self.get_neural_activity()

        # 用语言生成器训练
        loss = self.lang_gen.train_sequence(neural, target_text, reward_hint)

        # 用语言生成器生成输出
        output = self.lang_gen.generate_sequence(neural, max_length=40, temperature=0.7)

        # 计算输出奖励
        output_reward = self._compute_reward(output, target_text)
        total_reward = reward_hint * 0.3 + output_reward * 0.7

        # 注入脑奖励
        self.brain.inject_reward(total_reward)
        for _ in range(3):
            self.brain.train_language(total_reward)

        # 存储训练对用于回放
        self.lang_gen.store_training_pair(neural, target_text, total_reward)

        # 更新统计
        self.total_reward += total_reward
        self.reward_history.append(total_reward)
        self.loss_history.append(loss)
        self.turn_count += 1

        if verbose:
            match = '✓' if self._fuzzy_match(output, target_text) else '✗'
            print(f"  {match} '{input_text[:30]}' → '{output[:40]}' "
                  f"(expect: '{target_text[:30]}') loss={loss:.3f} r={total_reward:.3f}")

        return output, total_reward, loss

    def _compute_reward(self, output, target):
        """计算输出奖励"""
        if not output or output == '...':
            return -0.2

        out_lower = output.lower().strip()
        tgt_lower = target.lower().strip()

        if out_lower == tgt_lower:
            return 1.0

        out_chars = list(out_lower)
        tgt_chars = list(tgt_lower)

        # 字符级匹配率
        matches = 0
        for i in range(min(len(out_chars), len(tgt_chars))):
            if out_chars[i] == tgt_chars[i]:
                matches += 1
        if tgt_chars:
            char_acc = matches / len(tgt_chars)
            if char_acc > 0.5:
                return 0.3 + char_acc * 0.5
            elif char_acc > 0.2:
                return char_acc * 0.5

        # 词级匹配
        out_words = set(out_lower.split())
        tgt_words = set(tgt_lower.split())
        if tgt_words:
            overlap = len(out_words & tgt_words)
            if overlap > 0:
                return 0.2 + 0.1 * overlap

        return 0.0

    def _fuzzy_match(self, output, target, threshold=0.3):
        out_words = set(output.lower().split())
        tgt_words = set(target.lower().split())
        if not tgt_words:
            return False
        return len(out_words & tgt_words) / len(tgt_words) >= threshold

    def run_curriculum(self, curriculum=None, verbose=True):
        """运行课程训练"""
        if curriculum is None:
            curriculum = EXTENDED_CURRICULUM

        print("\n" + "=" * 60)
        print(f"SNA Enhanced Training - {len(curriculum)} levels")
        print(f"Neurons: {self.neuron_count:,}")
        print(f"Language generator: {self.lang_gen.vocab_size} vocab, "
              f"hidden={self.lang_gen.hidden_dim}")
        print("=" * 60)

        for level in curriculum:
            level_name = level['name']
            pairs = level['pairs']
            epochs = level.get('epochs', 50)
            base_reward = level.get('reward', 0.5)

            print(f"\n--- Level {level['level']}: {level_name} ({epochs} epochs) ---")

            level_rewards = []
            level_losses = []

            for epoch in range(epochs):
                random.shuffle(pairs)
                epoch_rewards = []
                epoch_losses = []

                for input_text, target_text in pairs:
                    output, reward, loss = self.train_pair(
                        input_text, target_text, base_reward, verbose=False)
                    epoch_rewards.append(reward)
                    epoch_losses.append(loss)

                level_rewards.extend(epoch_rewards)
                level_losses.extend(epoch_losses)

                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    avg_r = np.mean(epoch_rewards)
                    avg_l = np.mean(epoch_losses)
                    # 测试一个样本
                    test_in, test_out = pairs[0]
                    self.brain.inject_text(test_in)
                    for _ in range(15):
                        self.brain.step(0.0)
                    test_neural = self.get_neural_activity()
                    test_gen = self.lang_gen.generate_sequence(
                        test_neural, max_length=40, temperature=0.5)
                    print(f"  Epoch {epoch+1:>4}/{epochs}: "
                          f"avg_reward={avg_r:.3f} avg_loss={avg_l:.3f} | "
                          f"test: '{test_in}' → '{test_gen[:35]}'")

                # 记忆回放
                if epoch % 10 == 9:
                    replay_loss = self.lang_gen.replay_training(batch_size=20)
                    if verbose:
                        print(f"    [Replay] loss={replay_loss:.3f}")

            # 阶段总结
            avg_reward = np.mean(level_rewards)
            avg_loss = np.mean(level_losses)
            cs = self.brain.read_consciousness()
            print(f"  Level complete: avg_reward={avg_reward:.3f} "
                  f"avg_loss={avg_loss:.3f} Phi={cs.phi:.4f}")

            # 睡眠
            self.brain.sleep_cycle()

            # 保存
            self._save_checkpoint()

        self._print_report()

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 60)
        print("SNA Enhanced Interactive Mode")
        print("Commands: status, sleep, save, auto N, curriculum, quit")
        print("Type text to interact, +/- for feedback")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'status':
                self._print_status()
                continue
            if user_input.lower() == 'sleep':
                self.brain.sleep_cycle()
                print("Sleep complete.")
                continue
            if user_input.lower() == 'save':
                self._save_checkpoint()
                continue
            if user_input.lower() == 'curriculum':
                self.run_curriculum(verbose=True)
                continue
            if user_input.startswith('auto'):
                parts = user_input.split()
                n = int(parts[1]) if len(parts) > 1 else 100
                self._auto_train(n)
                continue

            if user_input == '+':
                self.brain.inject_reward(1.0)
                self.lang_gen.lr *= 1.1
                self.lang_gen.lr = min(0.01, self.lang_gen.lr)
                print("[+] Reward +1, lr boosted")
                continue
            if user_input == '-':
                self.brain.inject_reward(-0.5)
                self.lang_gen.lr *= 0.9
                print("[-] Reward -0.5, lr reduced")
                continue

            # 处理输入
            self.brain.inject_text(user_input)
            for _ in range(20):
                self.brain.step(0.0)
                self.total_steps += 1

            neural = self.get_neural_activity()
            output = self.lang_gen.generate_sequence(
                neural, max_length=60, temperature=0.6)

            cs = self.brain.read_consciousness()
            print(f"\n[SNA] {output}")
            print(f"  Phi={cs.phi:.4f} | Emotion={self.brain.get_emotion_label()} | "
                  f"Error={cs.self_prediction_error:.4f}")
            print(f"  Gen stats: lr={self.lang_gen.lr:.5f} "
                  f"pairs={len(self.lang_gen.training_pairs)}")

            self.turn_count += 1

            # 询问反馈
            try:
                feedback = input("  [Feedback: +/-/Enter] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if feedback == '+':
                reward = 0.8
                self.brain.inject_reward(reward)
                self.lang_gen.train_sequence(neural, output, reward)
                print("  [+] Trained with positive feedback")
            elif feedback == '-':
                try:
                    correct = input("  [Correct answer] ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if correct:
                    self.lang_gen.train_sequence(neural, correct, 0.8)
                    self.brain.inject_reward(-0.3)
                    print(f"  [-] Trained with correction: '{correct}'")
            else:
                # 自我训练
                self.lang_gen.train_sequence(neural, output, 0.1)

            # 定期回放
            if self.turn_count % 5 == 0:
                self.lang_gen.replay_training(batch_size=10)

            if self.turn_count % 30 == 0:
                self.brain.sleep_cycle()

        self._save_checkpoint()
        print("\nSession ended.")

    def _auto_train(self, n):
        """自动训练"""
        all_pairs = []
        for level in EXTENDED_CURRICULUM:
            for pair in level['pairs']:
                all_pairs.append((pair[0], pair[1], level.get('reward', 0.5)))

        print(f"\n[Auto] Training {n} turns...")
        for i in range(n):
            inp, tgt, rew = random.choice(all_pairs)
            output, reward, loss = self.train_pair(inp, tgt, rew, verbose=False)

            if i % 20 == 0:
                print(f"  Turn {i}/{n}: reward={reward:.3f} loss={loss:.3f} "
                      f"output='{output[:30]}'")

            # 回放
            if i % 10 == 0:
                self.lang_gen.replay_training(batch_size=10)

            # 睡眠
            if i % 100 == 99:
                self.brain.sleep_cycle()

        print(f"[Auto] Complete. {n} turns.")

    def _print_report(self):
        """打印最终报告"""
        cs = self.brain.read_consciousness()
        gen_stats = self.lang_gen.get_stats()

        print("\n" + "=" * 60)
        print("ENHANCED TRAINING REPORT")
        print("=" * 60)
        print(f"  Total steps:     {self.total_steps:,}")
        print(f"  Conversations:   {self.turn_count}")
        print(f"  Total reward:    {self.total_reward:.2f}")
        print(f"  Avg reward:      {np.mean(list(self.reward_history)):.3f}")
        print(f"  Phi:             {cs.phi:.4f}")
        print(f"  Ignition:        {cs.global_ignition:.4f}")
        print(f"  Self error:      {cs.self_prediction_error:.4f}")
        print(f"  Emotion:         {self.brain.get_emotion_label()}")
        print(f"\n  Language Generator:")
        print(f"    Vocab size:    {gen_stats['vocab_size']}")
        print(f"    Learning rate: {gen_stats['lr']:.6f}")
        print(f"    Training pairs:{gen_stats['training_pairs']}")
        print(f"    W1 norm:       {gen_stats['W1_norm']:.3f}")
        print(f"    W2 norm:       {gen_stats['W2_norm']:.3f}")
        print("=" * 60)

    def _print_status(self):
        cs = self.brain.read_consciousness()
        gen_stats = self.lang_gen.get_stats()
        print(f"\n  Steps: {self.total_steps:,} | Turns: {self.turn_count}")
        print(f"  Phi: {cs.phi:.4f} | Emotion: {self.brain.get_emotion_label()}")
        print(f"  Lang lr: {gen_stats['lr']:.5f} | pairs: {gen_stats['training_pairs']}")

    def _save_checkpoint(self):
        checkpoint = {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'turn_count': self.turn_count,
            'reward_history': list(self.reward_history)[-200:],
            'lang_stats': self.lang_gen.get_stats(),
            'timestamp': datetime.now().isoformat(),
        }
        path = os.path.join(self.save_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        # 保存语言模型权重
        np.savez(os.path.join(self.save_dir, 'lang_model.npz'),
                 W1=self.lang_gen.W1, b1=self.lang_gen.b1,
                 W2=self.lang_gen.W2, b2=self.lang_gen.b2)
        print(f"  [Save] Saved to {self.save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SNA Enhanced Trainer')
    parser.add_argument('-n', '--neurons', type=int, default=16000)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--auto', type=int, default=0)
    parser.add_argument('-i', '--interactive', action='store_true')
    args = parser.parse_args()

    trainer = EnhancedTrainer(neuron_count=args.neurons)

    if args.curriculum:
        trainer.run_curriculum(verbose=True)

    if args.auto > 0:
        trainer._auto_train(args.auto)

    if args.interactive or (not args.curriculum and args.auto == 0):
        trainer.interactive_mode()


if __name__ == '__main__':
    main()
