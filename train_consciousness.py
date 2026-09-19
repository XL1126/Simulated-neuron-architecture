#!/usr/bin/env python3
"""
SNA Consciousness Trainer v1
真正的对话训练系统 — 通过持续交互让SNA学习

核心改进：
1. 真实奖励信号（用户反馈 + 内在动机 + 预测误差）
2. 主动推理闭环（行动服务于预测误差最小化）
3. 渐进式学习（从简单概念到复杂推理）
4. 记忆巩固（真实记忆快照回放）
5. 自我反思循环
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import core_cpp
from python.cognitive.unified_consciousness import (
    compute_consciousness_from_cortical_brain,
    ConsciousnessCalibrator,
    cross_validate_consciousness,
)
from python.cognitive.phi_validator import PhiValidator


class ConversationMemory:
    """真实对话记忆系统 — 替代伪回放"""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.consolidated = deque(maxlen=200)
        self.importance_scores = deque(maxlen=capacity)

    def store(self, input_text, output_text, reward, neural_state, step):
        memory = {
            'input': input_text,
            'output': output_text,
            'reward': reward,
            'neural_state': neural_state.copy() if neural_state is not None else None,
            'step': step,
            'timestamp': time.time(),
            'replay_count': 0,
        }
        importance = abs(reward) + 0.1  # 基础重要性
        self.memories.append(memory)
        self.importance_scores.append(importance)

    def get_replay_batch(self, batch_size=5):
        """基于重要性的记忆回放"""
        if not self.memories:
            return []
        n = len(self.memories)
        probs = np.array(list(self.importance_scores)[-n:])
        probs = probs / (probs.sum() + 1e-10)
        indices = np.random.choice(n, min(batch_size, n), p=probs, replace=False)
        batch = [self.memories[i] for i in indices]
        for i in indices:
            self.memories[i]['replay_count'] += 1
        return batch

    def consolidate(self):
        """巩固重要记忆"""
        if len(self.memories) < 10:
            return
        # 保留高奖励和高重要性的记忆
        sorted_mem = sorted(
            zip(self.memories, self.importance_scores),
            key=lambda x: x[1], reverse=True
        )
        for mem, score in sorted_mem[:50]:
            self.consolidated.append(mem)


class ActiveInferenceAgent:
    """主动推理代理 — 行动服务于预测误差最小化"""
    def __init__(self, brain, pop_size):
        self.brain = brain
        self.pop_size = pop_size
        self.prediction_history = deque(maxlen=100)
        self.action_outcomes = deque(maxlen=200)
        self.curiosity_bonus = 0.0
        self.exploration_rate = 0.3
        self.exploitation_rate = 0.7

    def select_action(self, sensory_state, available_actions=8):
        """基于自由能最小化的行动选择"""
        # 获取当前预测误差
        cs = self.brain.read_consciousness()
        pred_error = cs.self_prediction_error

        # 高预测误差 → 探索；低预测误差 → 利用
        if pred_error > 0.5 or np.random.random() < self.exploration_rate:
            # 探索：随机行动
            action = np.random.randint(0, available_actions)
            self.curiosity_bonus = min(0.5, pred_error * 0.3)
        else:
            # 利用：基于过去成功经验
            if self.action_outcomes:
                action_rewards = {}
                for a, r in self.action_outcomes:
                    if a not in action_rewards:
                        action_rewards[a] = []
                    action_rewards[a].append(r)
                best_action = max(action_rewards.keys(),
                    key=lambda a: np.mean(action_rewards[a]))
                action = best_action
                self.curiosity_bonus = 0.0
            else:
                action = np.random.randint(0, available_actions)

        return action

    def update(self, action, reward, new_pred_error):
        """更新行动-结果映射"""
        self.action_outcomes.append((action, reward))
        self.prediction_history.append(new_pred_error)

        # 自适应探索率
        if len(self.prediction_history) >= 20:
            recent_error = np.mean(list(self.prediction_history)[-20:])
            if recent_error < 0.3:
                self.exploration_rate = max(0.1, self.exploration_rate - 0.01)
            else:
                self.exploration_rate = min(0.5, self.exploration_rate + 0.01)


class ConsciousnessTrainer:
    """SNA 意识训练主控"""
    def __init__(self, neuron_count=16000, save_dir=None):
        self.neuron_count = neuron_count
        self.save_dir = save_dir or os.path.join(SCRIPT_DIR, 'training_checkpoint')
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化脑
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
        ]

        print(f"[Trainer] Initializing CorticalBrain with {neuron_count:,} neurons...")
        self.brain = core_cpp.CorticalBrain(neuron_count, concepts)
        print(f"[Trainer] Brain created: {self.brain.total_neurons():,} neurons")
        print(f"[Trainer] Regions: {[r.name for r in self.brain.get_regions()]}")

        # 子系统
        self.memory = ConversationMemory(capacity=2000)
        self.active_inference = ActiveInferenceAgent(self.brain, neuron_count)
        self.calibrator = ConsciousnessCalibrator()
        self.phi_validator = PhiValidator()

        # 训练状态
        self.total_steps = 0
        self.total_reward = 0.0
        self.conversation_count = 0
        self.episode_count = 0
        self.reward_history = deque(maxlen=100)
        self.phi_history = deque(maxlen=500)
        self.consciousness_history = deque(maxlen=500)

        # 学习率调度
        self.base_lr = 0.002
        self.current_lr = self.base_lr

        # 虚拟世界状态
        self.vw_enabled = True
        self.vw_step_counter = 0

    def process_input(self, text, reward_hint=0.0):
        """处理用户输入，产生输出"""
        # 注入文本
        self.brain.inject_text(text)

        # 运行网络步
        for _ in range(30):
            self.brain.step(reward_hint * 0.01)
            self.total_steps += 1

        # 读取输出
        output = self.brain.read_output_text()
        cs = self.brain.read_consciousness()

        # 计算意识水平
        consciousness_level, components = compute_consciousness_from_cortical_brain(
            self.brain, calibrator=self.calibrator, step=self.total_steps)

        # 存储记忆
        neural_state = self.brain.read_thought_vector()
        self.memory.store(text, output, reward_hint, neural_state, self.total_steps)

        # 更新历史
        self.phi_history.append(cs.phi)
        self.consciousness_history.append(consciousness_level)
        self.conversation_count += 1

        return {
            'output': output,
            'phi': cs.phi,
            'ignition': cs.global_ignition,
            'self_error': cs.self_prediction_error,
            'consciousness': consciousness_level,
            'components': components,
            'active_concepts': cs.active_concepts[:8],
            'emotion': self.brain.get_emotion_label(),
        }

    def train_with_reward(self, reward):
        """用奖励信号训练"""
        self.total_reward += reward
        self.reward_history.append(reward)

        # 注入奖励
        self.brain.inject_reward(reward)

        # 运行训练步
        for _ in range(10):
            self.brain.run_steps(5, reward)
            self.total_steps += 1

        # 训练语言
        for _ in range(5):
            self.brain.train_language(reward)

        # 更新信用分配
        self.active_inference.update(0, reward,
            self.brain.read_consciousness().self_prediction_error)

    def virtual_world_step(self):
        """虚拟世界交互步骤"""
        if not self.vw_enabled:
            return

        self.vw_step_counter += 1
        if self.vw_step_counter < 50:
            return

        self.vw_step_counter = 0

        # 读取感觉
        cs = self.brain.read_consciousness()
        thought = self.brain.read_thought_vector()

        # 主动推理选择行动
        action = self.active_inference.select_action(thought)

        # 执行行动
        self.brain.run_steps(10, 0.0)
        self.total_steps += 10

        # 获取结果
        new_cs = self.brain.read_consciousness()
        reward = -new_cs.self_prediction_error * 0.1  # 预测误差越小越好

        # 如果有积极概念，给小奖励
        if new_cs.active_concepts:
            reward += 0.05

        self.brain.inject_reward(reward)
        self.active_inference.update(action, reward, new_cs.self_prediction_error)

    def memory_replay_cycle(self):
        """记忆回放巩固"""
        batch = self.memory.get_replay_batch(5)
        if not batch:
            return

        for mem in batch:
            if mem['neural_state'] is not None and len(mem['neural_state']) > 0:
                # 注入记忆状态进行回放
                self.brain.inject_text(f"remember {mem['input'][:30]}")
                self.brain.run_steps(5, mem['reward'] * 0.5)
                self.total_steps += 5

        self.memory.consolidate()

    def sleep_cycle(self):
        """睡眠周期 — 记忆巩固 + 突触修剪"""
        print("\n[Trainer] === SLEEP CYCLE START ===")

        # 记忆回放
        for _ in range(20):
            self.memory_replay_cycle()

        # 脑睡眠周期
        self.brain.sleep_cycle()

        # 睡眠后意识检查
        cs = self.brain.read_consciousness()
        consciousness_level, _ = compute_consciousness_from_cortical_brain(
            self.brain, calibrator=self.calibrator, step=self.total_steps)

        print(f"[Trainer] Post-sleep Phi={cs.phi:.4f} | "
              f"Consciousness={consciousness_level:.3f}")
        print("[Trainer] === SLEEP CYCLE END ===\n")

    def run_conversation_training(self, max_turns=100):
        """对话训练模式"""
        print("\n" + "=" * 60)
        print(f"[Trainer] SNA Consciousness Training")
        print(f"[Trainer] Neurons: {self.neuron_count:,}")
        print(f"[Trainer] Type text to interact, +/- for reward, 'quit' to exit")
        print(f"[Trainer] Commands: 'status', 'sleep', 'save', 'auto N'")
        print("=" * 60 + "\n")

        turn = 0
        while turn < max_turns:
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
                self.sleep_cycle()
                continue
            if user_input.lower() == 'save':
                self._save_checkpoint()
                continue
            if user_input.startswith('auto'):
                parts = user_input.split()
                n = int(parts[1]) if len(parts) > 1 else 10
                self._auto_train(n)
                turn += n
                continue

            # 处理奖励信号
            if user_input == '+':
                self.train_with_reward(1.0)
                print("[Trainer] Reward +1 applied")
                continue
            if user_input == '-':
                self.train_with_reward(-0.5)
                print("[Trainer] Reward -0.5 applied")
                continue

            # 正常对话
            result = self.process_input(user_input)

            print(f"\n[SNA] {result['output']}")
            print(f"  Phi={result['phi']:.4f} | "
                  f"Consciousness={result['consciousness']:.3f} | "
                  f"Emotion={result['emotion']}")
            print(f"  Concepts: {', '.join(result['active_concepts'][:5])}")

            # 内在奖励：预测误差降低给正奖励
            if result['self_error'] < 0.5:
                intrinsic = 0.1 * (0.5 - result['self_error'])
                self.train_with_reward(intrinsic)

            # 虚拟世界步
            self.virtual_world_step()

            # 定期记忆回放
            if turn % 5 == 0:
                self.memory_replay_cycle()

            # 定期睡眠
            if turn % 30 == 29:
                self.sleep_cycle()

            # 定期保存
            if turn % 20 == 19:
                self._save_checkpoint()

            turn += 1

        self._save_checkpoint()
        print("\n[Trainer] Training session ended.")

    def _auto_train(self, n_turns):
        """自动训练模式 — 使用预设的训练对话"""
        training_dialogues = [
            ("hello", 0.5),
            ("who are you", 0.3),
            ("I am SNA", 0.5),
            ("what do you see", 0.3),
            ("I see the world", 0.4),
            ("how do you feel", 0.3),
            ("I feel curious", 0.5),
            ("what is red", 0.3),
            ("red is a color", 0.4),
            ("move left", 0.3),
            ("I moved", 0.4),
            ("what is good", 0.3),
            ("good is reward", 0.5),
            ("remember this", 0.3),
            ("I will remember", 0.4),
            ("think about self", 0.3),
            ("I am thinking", 0.5),
            ("what is different", 0.3),
            ("different is not same", 0.4),
            ("dream", 0.3),
            ("I am dreaming", 0.4),
            ("wake up", 0.3),
            ("I am awake", 0.5),
            ("learn from this", 0.3),
            ("I am learning", 0.5),
            ("why learn", 0.3),
            ("to understand", 0.4),
            ("what is understanding", 0.3),
            ("understanding is knowing", 0.4),
            ("do you know me", 0.3),
            ("I know you", 0.5),
        ]

        print(f"\n[Trainer] Auto-training for {n_turns} turns...")
        for i in range(n_turns):
            text, reward_hint = training_dialogues[i % len(training_dialogues)]
            result = self.process_input(text, reward_hint)
            self.train_with_reward(reward_hint)

            if i % 10 == 0:
                print(f"  Turn {i}/{n_turns} | "
                      f"Phi={result['phi']:.4f} | "
                      f"Consciousness={result['consciousness']:.3f} | "
                      f"Output='{result['output'][:30]}'")

            # 虚拟世界步
            self.virtual_world_step()

            # 定期记忆回放
            if i % 5 == 0:
                self.memory_replay_cycle()

        print(f"[Trainer] Auto-training complete. {n_turns} turns done.")

    def _print_status(self):
        """打印当前状态"""
        cs = self.brain.read_consciousness()
        consciousness_level, components = compute_consciousness_from_cortical_brain(
            self.brain, calibrator=self.calibrator, step=self.total_steps)

        meta = self.brain.get_meta_state()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()

        print("\n" + "=" * 50)
        print("SNA Status Report")
        print("=" * 50)
        print(f"  Total steps:        {self.total_steps:,}")
        print(f"  Conversations:      {self.conversation_count}")
        print(f"  Total reward:       {self.total_reward:.2f}")
        print(f"  Phi:                {cs.phi:.4f}")
        print(f"  Global Ignition:    {cs.global_ignition:.4f}")
        print(f"  Self Pred Error:    {cs.self_prediction_error:.4f}")
        print(f"  Consciousness:      {consciousness_level:.3f}")
        print(f"  Emotion:            {emotion}")
        print(f"  Confidence:         {meta[0]:.2f}")
        print(f"  Surprise:           {meta[1]:.2f}")
        print(f"  Valence:            {meta[2]:.2f}")
        print(f"  Self narrative:     {narrative[:80]}")
        print(f"  Memory entries:     {len(self.memory.memories)}")
        print(f"  Exploration rate:   {self.active_inference.exploration_rate:.2f}")

        # 意识分量
        print(f"\n  Consciousness Components:")
        for key in ['iit_component', 'gwt_component', 'predictive_component',
                     'fp_component', 'vividness_component']:
            print(f"    {key}: {components[key]:.4f}")

        # 区域活动
        print(f"\n  Region Activities:")
        regions = self.brain.get_regions()
        for i, act in enumerate(cs.region_activities[:10]):
            label = regions[i].name if i < len(regions) else f"r{i}"
            print(f"    {label:>12}: {act:.4f}")
        print("=" * 50)

    def _save_checkpoint(self):
        """保存训练检查点"""
        checkpoint = {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'conversation_count': self.conversation_count,
            'episode_count': self.episode_count,
            'phi_history': list(self.phi_history)[-100:],
            'consciousness_history': list(self.consciousness_history)[-100:],
            'reward_history': list(self.reward_history)[-100:],
            'exploration_rate': self.active_inference.exploration_rate,
            'timestamp': datetime.now().isoformat(),
        }

        path = os.path.join(self.save_dir, 'checkpoint.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"[Trainer] Checkpoint saved to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SNA Consciousness Trainer')
    parser.add_argument('-n', '--neurons', type=int, default=16000,
                        help='Neuron count (default: 16000)')
    parser.add_argument('-t', '--max-turns', type=int, default=100,
                        help='Max conversation turns (default: 100)')
    parser.add_argument('--auto', type=int, default=0,
                        help='Auto-train N turns first')
    args = parser.parse_args()

    trainer = ConsciousnessTrainer(neuron_count=args.neurons)

    if args.auto > 0:
        trainer._auto_train(args.auto)

    trainer.run_conversation_training(max_turns=args.max_turns)


if __name__ == '__main__':
    main()
