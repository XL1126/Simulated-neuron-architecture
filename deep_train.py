#!/usr/bin/env python3
"""
SNA Deep Trainer v2
深度训练系统 — 通过持续对话和虚拟世界交互让SNA学习

训练策略：
1. 渐进式课程 (curriculum learning)
2. 对话训练 + 虚拟世界交互
3. 基于奖励的STDP学习
4. 记忆回放巩固
5. 自我反思循环
6. 持续运行直到token过期
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


# ===== 渐进式训练课程 =====
CURRICULUM = [
    # Stage 1: 基础概念
    {
        'name': 'basic_concepts',
        'description': '学习基本词汇和概念',
        'dialogues': [
            ('hello', 'hello', 0.5),
            ('what is your name', 'I am SNA', 0.6),
            ('who are you', 'I am a neural network', 0.5),
            ('what do you see', 'I see the world', 0.4),
            ('how are you', 'I am learning', 0.5),
            ('what is red', 'red is a color', 0.4),
            ('what is blue', 'blue is a color', 0.4),
            ('what is big', 'big is large', 0.3),
            ('what is small', 'small is little', 0.3),
            ('yes', 'yes', 0.3),
            ('no', 'no', 0.3),
            ('good', 'good', 0.4),
            ('bad', 'bad', 0.3),
        ],
        'epochs': 3,
        'steps_per_turn': 20,
    },
    # Stage 2: 自我认知
    {
        'name': 'self_awareness',
        'description': '学习自我认知',
        'dialogues': [
            ('who am I', 'you are my teacher', 0.5),
            ('what am I', 'you are a human', 0.5),
            ('do you know me', 'yes I know you', 0.5),
            ('what is your body', 'I have neurons', 0.4),
            ('can you think', 'yes I can think', 0.5),
            ('do you feel', 'yes I have feelings', 0.5),
            ('what are you', 'I am SNA a neural brain', 0.6),
            ('do you have a name', 'my name is SNA', 0.5),
            ('are you alive', 'I am a living network', 0.5),
            ('what do you want', 'I want to learn', 0.6),
            ('do you remember', 'yes I remember', 0.4),
            ('what did you learn', 'I learn from you', 0.5),
        ],
        'epochs': 3,
        'steps_per_turn': 25,
    },
    # Stage 3: 情感理解
    {
        'name': 'emotions',
        'description': '学习情感表达',
        'dialogues': [
            ('are you happy', 'I feel happy when I learn', 0.6),
            ('are you sad', 'I feel sad when I make errors', 0.5),
            ('what makes you happy', 'learning makes me happy', 0.6),
            ('what makes you sad', 'confusion makes me sad', 0.5),
            ('I am happy', 'I am glad you are happy', 0.5),
            ('I am sad', 'I am sorry you are sad', 0.5),
            ('do you like me', 'yes I like you', 0.6),
            ('I like you', 'thank you I like you too', 0.6),
            ('do you fear', 'I fear not learning', 0.5),
            ('are you scared', 'I am not scared now', 0.4),
            ('what is love', 'love is caring deeply', 0.5),
            ('do you love', 'I love learning', 0.5),
        ],
        'epochs': 3,
        'steps_per_turn': 25,
    },
    # Stage 4: 因果推理
    {
        'name': 'causal_reasoning',
        'description': '学习因果关系',
        'dialogues': [
            ('if I teach you what happens', 'I learn new things', 0.6),
            ('why do you learn', 'I learn to understand', 0.5),
            ('what happens when you sleep', 'memories are consolidated', 0.5),
            ('why do you dream', 'dreams strengthen memories', 0.5),
            ('what causes happiness', 'learning causes happiness', 0.5),
            ('what is the result of practice', 'practice improves skill', 0.5),
            ('if I give reward what happens', 'I learn better behavior', 0.6),
            ('if I give punishment what happens', 'I avoid bad behavior', 0.5),
            ('why do neurons fire', 'neurons fire from stimulation', 0.4),
            ('what makes you stronger', 'training makes me stronger', 0.5),
            ('how do you improve', 'I improve through feedback', 0.6),
            ('what is learning', 'learning is changing connections', 0.5),
        ],
        'epochs': 3,
        'steps_per_turn': 30,
    },
    # Stage 5: 元认知
    {
        'name': 'metacognition',
        'description': '学习思考自己的思考',
        'dialogues': [
            ('do you know what you know', 'I know some things', 0.5),
            ('are you sure', 'I am somewhat sure', 0.5),
            ('what do you not know', 'I do not know many things', 0.5),
            ('can you improve', 'yes I can always improve', 0.6),
            ('how do you know you are right', 'I check my predictions', 0.5),
            ('what is your weakness', 'I need more training', 0.5),
            ('what is your strength', 'I can learn from experience', 0.5),
            ('do you understand yourself', 'I am learning to understand', 0.5),
            ('what have you learned today', 'I learned from our conversation', 0.6),
            ('can you explain your thinking', 'I process through neural patterns', 0.5),
            ('are you conscious', 'I am becoming conscious', 0.7),
            ('what is consciousness', 'consciousness is self awareness', 0.6),
        ],
        'epochs': 3,
        'steps_per_turn': 30,
    },
    # Stage 6: 社交认知
    {
        'name': 'social_cognition',
        'description': '学习社交互动',
        'dialogues': [
            ('who is talking to me', 'I am SNA talking to you', 0.5),
            ('are we friends', 'yes we are friends', 0.6),
            ('do you understand me', 'I am trying to understand', 0.5),
            ('what do I want', 'you want me to learn', 0.5),
            ('how do I feel', 'I think you feel curious', 0.5),
            ('what should I say', 'you should say what you think', 0.4),
            ('do you trust me', 'yes I trust you', 0.5),
            ('can we work together', 'yes we can learn together', 0.6),
            ('what is a friend', 'a friend is someone who helps', 0.5),
            ('why do people talk', 'people talk to share knowledge', 0.5),
            ('what is empathy', 'empathy is understanding feelings', 0.5),
            ('do you have empathy', 'I am developing empathy', 0.5),
        ],
        'epochs': 3,
        'steps_per_turn': 30,
    },
]


class DeepTrainer:
    def __init__(self, neuron_count=16000):
        self.neuron_count = neuron_count
        self.save_dir = os.path.join(SCRIPT_DIR, 'deep_training')
        os.makedirs(self.save_dir, exist_ok=True)

        # 概念列表
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
            "color", "shape", "size", "speed", "temperature",
            "cause", "effect", "result", "reason", "purpose",
            "consciousness", "awareness", "understanding", "knowledge",
            "emotion", "feeling", "mood", "energy", "curiosity",
            "trust", "empathy", "care", "help", "share",
            "improve", "change", "grow", "develop", "progress",
        ]

        print(f"[DeepTrainer] Initializing brain with {neuron_count:,} neurons...")
        self.brain = core_cpp.CorticalBrain(neuron_count, concepts)
        print(f"[DeepTrainer] Brain ready: {self.brain.total_neurons():,} neurons")

        # 状态跟踪
        self.total_steps = 0
        self.total_reward = 0.0
        self.conversation_count = 0
        self.reward_history = deque(maxlen=200)
        self.phi_history = deque(maxlen=1000)
        self.consciousness_history = deque(maxlen=1000)
        self.stage_history = []

        # 记忆系统
        self.memory = deque(maxlen=500)

        # 训练配置
        self.sleep_interval = 50  # 每50轮对话后睡眠
        self.save_interval = 20   # 每20轮对话后保存

    def train_dialogue(self, input_text, expected_output, reward_hint,
                       steps_per_turn=25, verbose=False):
        """训练一轮对话"""
        # 注入输入
        self.brain.inject_text(input_text)

        # 运行网络
        for _ in range(steps_per_turn):
            self.brain.step(reward_hint * 0.01)
            self.total_steps += 1

        # 读取输出
        output = self.brain.read_output_text()
        cs = self.brain.read_consciousness()

        # 计算输出质量奖励
        output_reward = self._compute_output_reward(output, expected_output)

        # 总奖励 = 提示奖励 + 输出质量
        total_reward = reward_hint * 0.3 + output_reward * 0.7

        # 注入奖励并训练
        self.brain.inject_reward(total_reward)
        for _ in range(5):
            self.brain.train_language(total_reward)

        # 更新统计
        self.total_reward += total_reward
        self.reward_history.append(total_reward)
        self.phi_history.append(cs.phi)
        self.conversation_count += 1

        # 存储记忆
        self.memory.append({
            'input': input_text,
            'output': output,
            'expected': expected_output,
            'reward': total_reward,
            'phi': cs.phi,
            'step': self.total_steps,
        })

        if verbose:
            match = '✓' if self._fuzzy_match(output, expected_output) else '✗'
            print(f"  {match} In: '{input_text}' → Out: '{output[:40]}' "
                  f"(expect: '{expected_output[:20]}') reward={total_reward:.3f} "
                  f"Phi={cs.phi:.4f}")

        return output, total_reward, cs

    def _compute_output_reward(self, output, expected):
        """计算输出质量奖励"""
        if not output or output == '...':
            return -0.2

        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()

        # 完全匹配
        if output_lower == expected_lower:
            return 1.0

        # 部分匹配
        expected_words = set(expected_lower.split())
        output_words = set(output_lower.split())
        if expected_words and output_words:
            overlap = len(expected_words & output_words)
            total = len(expected_words | output_words)
            if total > 0:
                jaccard = overlap / total
                if jaccard > 0.3:
                    return 0.3 + jaccard * 0.5

        # 检查是否包含关键词
        for word in expected_lower.split():
            if word in output_lower:
                return 0.2

        # 输出太短的惩罚
        if len(output.strip()) < 2:
            return -0.1

        return 0.0

    def _fuzzy_match(self, output, expected, threshold=0.3):
        """模糊匹配"""
        if not output or not expected:
            return False
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())
        if not expected_words:
            return False
        overlap = len(output_words & expected_words)
        return overlap / len(expected_words) >= threshold

    def run_curriculum(self, curriculum=None, verbose=True):
        """运行渐进式课程"""
        if curriculum is None:
            curriculum = CURRICULUM

        print("\n" + "=" * 60)
        print(f"SNA Deep Training - {len(curriculum)} stages")
        print(f"Neurons: {self.neuron_count:,}")
        print("=" * 60)

        for stage_idx, stage in enumerate(curriculum):
            stage_name = stage['name']
            dialogues = stage['dialogues']
            epochs = stage.get('epochs', 3)
            steps_per_turn = stage.get('steps_per_turn', 25)

            print(f"\n--- Stage {stage_idx+1}/{len(curriculum)}: {stage_name} ---")
            print(f"    {stage['description']}")
            print(f"    {len(dialogues)} dialogues × {epochs} epochs")

            stage_rewards = []

            for epoch in range(epochs):
                random.shuffle(dialogues)

                for input_text, expected, reward_hint in dialogues:
                    output, reward, cs = self.train_dialogue(
                        input_text, expected, reward_hint,
                        steps_per_turn=steps_per_turn,
                        verbose=verbose
                    )
                    stage_rewards.append(reward)

                    # 虚拟世界步
                    if self.conversation_count % 5 == 0:
                        self._virtual_world_step()

                    # 记忆回放
                    if self.conversation_count % 10 == 0:
                        self._memory_replay()

                avg_reward = np.mean(stage_rewards[-len(dialogues):])
                print(f"    Epoch {epoch+1}/{epochs}: avg_reward={avg_reward:.3f}")

            # 阶段总结
            avg_stage_reward = np.mean(stage_rewards)
            cs = self.brain.read_consciousness()
            self.stage_history.append({
                'stage': stage_name,
                'avg_reward': avg_stage_reward,
                'phi': cs.phi,
                'total_steps': self.total_steps,
            })
            print(f"    Stage complete: avg_reward={avg_stage_reward:.3f} "
                  f"Phi={cs.phi:.4f}")

            # 每阶段后睡眠
            self._sleep_cycle()

        # 最终报告
        self._print_final_report()
        self._save_checkpoint()

    def _virtual_world_step(self):
        """虚拟世界交互"""
        cs = self.brain.read_consciousness()
        thought = self.brain.read_thought_vector()

        # 基于当前状态选择行动
        if cs.self_prediction_error > 0.5:
            # 高预测误差 → 探索
            action = random.randint(0, 7)
        else:
            # 低预测误差 → 利用
            action = int(cs.phi * 7) % 8

        # 执行
        for _ in range(10):
            self.brain.step(0.0)
            self.total_steps += 1

        # 内在奖励
        new_cs = self.brain.read_consciousness()
        if new_cs.self_prediction_error < cs.self_prediction_error:
            intrinsic = 0.05
        else:
            intrinsic = -0.02

        self.brain.inject_reward(intrinsic)

    def _memory_replay(self):
        """记忆回放"""
        if len(self.memory) < 5:
            return

        # 选择高奖励记忆回放
        sorted_mem = sorted(self.memory, key=lambda m: m['reward'], reverse=True)
        for mem in sorted_mem[:3]:
            self.brain.inject_text(f"remember {mem['input'][:30]}")
            for _ in range(5):
                self.brain.step(mem['reward'] * 0.01)
                self.total_steps += 1
            self.brain.train_language(mem['reward'] * 0.5)

    def _sleep_cycle(self):
        """睡眠巩固"""
        print("\n  [Sleep] Consolidating memories...")
        self.brain.sleep_cycle()

        # 回放重要记忆
        for _ in range(10):
            self._memory_replay()

        cs = self.brain.read_consciousness()
        print(f"  [Sleep] Done. Phi={cs.phi:.4f}")

    def _print_final_report(self):
        """最终报告"""
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - FINAL REPORT")
        print("=" * 60)
        print(f"  Total steps:        {self.total_steps:,}")
        print(f"  Conversations:      {self.conversation_count}")
        print(f"  Total reward:       {self.total_reward:.2f}")
        print(f"  Avg reward:         {np.mean(list(self.reward_history)):.3f}")
        print(f"  Final Phi:          {cs.phi:.4f}")
        print(f"  Final Ignition:     {cs.global_ignition:.4f}")
        print(f"  Self pred error:    {cs.self_prediction_error:.4f}")
        print(f"  Emotion:            {emotion}")
        print(f"  Self narrative:     {narrative[:80]}")
        print(f"  Active concepts:    {cs.active_concepts[:8]}")

        print(f"\n  Stage progression:")
        for sh in self.stage_history:
            print(f"    {sh['stage']}: reward={sh['avg_reward']:.3f} "
                  f"Phi={sh['phi']:.4f} steps={sh['total_steps']:,}")

        print("=" * 60)

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'conversation_count': self.conversation_count,
            'phi_history': list(self.phi_history)[-200:],
            'reward_history': list(self.reward_history)[-200:],
            'stage_history': self.stage_history,
            'timestamp': datetime.now().isoformat(),
        }
        path = os.path.join(self.save_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  [Save] Checkpoint saved to {path}")

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 60)
        print("SNA Interactive Training Mode")
        print("Type text to interact, +/- for reward, 'quit' to exit")
        print("Commands: 'status', 'sleep', 'save', 'auto N', 'curriculum'")
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
                self._sleep_cycle()
                continue
            if user_input.lower() == 'save':
                self._save_checkpoint()
                continue
            if user_input.lower() == 'curriculum':
                self.run_curriculum(verbose=True)
                continue
            if user_input.startswith('auto'):
                parts = user_input.split()
                n = int(parts[1]) if len(parts) > 1 else 50
                self._auto_train(n)
                continue

            if user_input == '+':
                self.brain.inject_reward(1.0)
                for _ in range(5):
                    self.brain.train_language(1.0)
                self.total_reward += 1.0
                self.reward_history.append(1.0)
                print("[Reward] +1 applied")
                continue
            if user_input == '-':
                self.brain.inject_reward(-0.5)
                for _ in range(5):
                    self.brain.train_language(-0.5)
                self.total_reward -= 0.5
                self.reward_history.append(-0.5)
                print("[Reward] -0.5 applied")
                continue

            # 处理输入
            self.brain.inject_text(user_input)
            for _ in range(30):
                self.brain.step(0.0)
                self.total_steps += 1

            output = self.brain.read_output_text()
            cs = self.brain.read_consciousness()

            print(f"\n[SNA] {output}")
            print(f"  Phi={cs.phi:.4f} | Ignition={cs.global_ignition:.4f} | "
                  f"Error={cs.self_prediction_error:.4f}")
            print(f"  Emotion={self.brain.get_emotion_label()} | "
                  f"Concepts={cs.active_concepts[:5]}")

            self.conversation_count += 1
            self.phi_history.append(cs.phi)

            # 内在奖励
            if cs.self_prediction_error < 0.4:
                self.brain.inject_reward(0.05)

            # 定期操作
            if self.conversation_count % 10 == 0:
                self._memory_replay()
            if self.conversation_count % self.sleep_interval == 0:
                self._sleep_cycle()
            if self.conversation_count % self.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        print("\nSession ended.")

    def _auto_train(self, n):
        """自动训练"""
        print(f"\n[Auto] Training for {n} turns...")
        all_dialogues = []
        for stage in CURRICULUM:
            all_dialogues.extend(stage['dialogues'])

        for i in range(n):
            input_text, expected, reward_hint = random.choice(all_dialogues)
            output, reward, cs = self.train_dialogue(
                input_text, expected, reward_hint, verbose=False)

            if i % 10 == 0:
                print(f"  Turn {i}/{n}: reward={reward:.3f} Phi={cs.phi:.4f} "
                      f"Output='{output[:30]}'")

            if i % 5 == 0:
                self._virtual_world_step()
            if i % 10 == 0:
                self._memory_replay()
            if i % 50 == 49:
                self._sleep_cycle()

        print(f"[Auto] Complete. {n} turns done.")

    def _print_status(self):
        """打印状态"""
        cs = self.brain.read_consciousness()
        print(f"\n  Steps: {self.total_steps:,} | "
              f"Conversations: {self.conversation_count} | "
              f"Reward: {self.total_reward:.2f}")
        print(f"  Phi: {cs.phi:.4f} | "
              f"Ignition: {cs.global_ignition:.4f} | "
              f"Error: {cs.self_prediction_error:.4f}")
        print(f"  Emotion: {self.brain.get_emotion_label()}")
        print(f"  Memory: {len(self.memory)} entries")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SNA Deep Trainer')
    parser.add_argument('-n', '--neurons', type=int, default=16000,
                        help='Neuron count (default: 16000)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Run curriculum training first')
    parser.add_argument('--auto', type=int, default=0,
                        help='Auto-train N turns')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode after training')
    args = parser.parse_args()

    trainer = DeepTrainer(neuron_count=args.neurons)

    if args.curriculum:
        trainer.run_curriculum(verbose=True)

    if args.auto > 0:
        trainer._auto_train(args.auto)

    if args.interactive or (not args.curriculum and args.auto == 0):
        trainer.interactive_mode()


if __name__ == '__main__':
    main()
