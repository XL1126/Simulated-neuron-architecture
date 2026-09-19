#!/usr/bin/env python3
"""
SNA Neural Language Generator
基于神经活动的语言生成系统 — 替代原来的余弦相似度查找表

核心思路：
1. 神经活动 → 编码为连续向量
2. 向量通过学习到的权重映射到字符概率
3. 通过训练逐步建立"神经模式→语言"的映射
4. 支持上下文记忆（前文影响后续生成）
"""

import numpy as np
from collections import deque


class NeuralLanguageGenerator:
    def __init__(self, neural_dim=256, vocab_size=200, hidden_dim=128,
                 context_window=5):
        self.neural_dim = neural_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_window = context_window

        # 字符映射
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocab()

        # 神经网络权重（简单3层）— 延迟初始化，等 _build_vocab 后确定实际维度
        self._w1_input_dim = None  # 会在 _ensure_weights 中计算
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # 上下文缓冲
        self.context_buffer = deque(maxlen=context_window)
        self._reset_context()

        # 学习率
        self.lr = 0.001
        self.lr_decay = 0.9999

        # 训练数据缓冲
        self.training_pairs = deque(maxlen=5000)

        # 激活历史（用于反向传播）
        self._last_hidden = None
        self._last_output = None
        self._last_input = None

    def _build_vocab(self):
        """构建字符词汇表"""
        # 基本ASCII
        idx = 0
        for i in range(32, 127):
            self.char_to_idx[chr(i)] = idx
            self.idx_to_char[idx] = chr(i)
            idx += 1

        # 常用中文标点
        for ch in ['，', '。', '！', '？', '、', '…', '：', '；',
                    '"', '"', ''', ''', '（', '）', '【', '】']:
            if idx < self.vocab_size:
                self.char_to_idx[ch] = idx
                self.idx_to_char[idx] = ch
                idx += 1

        # 特殊token
        for token in ['<PAD>', '<START>', '<END>', '<UNK>']:
            if idx < self.vocab_size:
                self.char_to_idx[token] = idx
                self.idx_to_char[idx] = token
                idx += 1

        self.vocab_size = min(self.vocab_size, idx)

    def _reset_context(self):
        """重置上下文"""
        self.context_buffer.clear()
        for _ in range(self.context_window):
            self.context_buffer.append(np.zeros(self.vocab_size, dtype=np.float32))

    def _build_input(self, neural_activity):
        """构建输入向量：神经活动 + 上下文"""
        neural = np.array(neural_activity, dtype=np.float32).flatten()
        if len(neural) > self.neural_dim:
            neural = neural[:self.neural_dim]
        elif len(neural) < self.neural_dim:
            neural = np.pad(neural, (0, self.neural_dim - len(neural)))

        context = np.concatenate(list(self.context_buffer))
        return np.concatenate([neural, context])

    def _ensure_weights(self):
        """确保权重已初始化（延迟到 vocab_size 确定后）"""
        if self._w1_input_dim is not None:
            return
        input_dim = self.neural_dim + self.vocab_size * self.context_window
        self.W1 = np.random.normal(0, 0.02, (self.hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = np.random.normal(0, 0.02, (self.vocab_size, self.hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(self.vocab_size, dtype=np.float32)
        self._w1_input_dim = input_dim

    def forward(self, neural_activity):
        """前向传播：神经活动 → 字符概率"""
        self._ensure_weights()
        x = self._build_input(neural_activity)

        # 隐藏层
        hidden = self.W1 @ x + self.b1
        hidden = np.maximum(0, hidden)  # ReLU

        # 输出层
        logits = self.W2 @ hidden + self.b2
        probs = self._softmax(logits)

        # 保存用于反向传播
        self._last_hidden = hidden
        self._last_output = probs
        self._last_input = x

        return probs

    def generate_char(self, neural_activity, temperature=0.8, top_k=10):
        """生成一个字符"""
        probs = self.forward(neural_activity)

        # 温度缩放
        if temperature > 0:
            logits = np.log(probs + 1e-10) / temperature
            probs = self._softmax(logits)

        # Top-k 采样
        if top_k > 0 and top_k < len(probs):
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / (top_probs.sum() + 1e-10)
            idx = np.random.choice(top_indices, p=top_probs)
        else:
            idx = np.random.choice(len(probs), p=probs)

        # 更新上下文
        one_hot = np.zeros(self.vocab_size, dtype=np.float32)
        one_hot[idx] = 1.0
        self.context_buffer.append(one_hot)

        char = self.idx_to_char.get(idx, '?')
        return char, float(probs[idx])

    def generate_sequence(self, neural_activity, max_length=50,
                          temperature=0.8, end_chars='.!?\n'):
        """生成一个序列"""
        self._reset_context()
        chars = []

        for _ in range(max_length):
            char, conf = self.generate_char(neural_activity, temperature)
            if char == '<END>':
                break
            if char in ('<PAD>', '<START>'):
                continue
            chars.append(char)
            if char in end_chars and len(chars) > 3:
                break

        return ''.join(chars)

    def train_step(self, neural_activity, target_char, reward=0.0):
        """一步训练：给定神经活动和目标字符"""
        probs = self.forward(neural_activity)

        target_idx = self.char_to_idx.get(target_char, 0)

        # 交叉熵梯度 + 奖励调制
        grad_output = probs.copy()
        grad_output[target_idx] -= 1.0

        # 奖励调制：正奖励增强目标概率，负奖励减弱
        if reward != 0:
            grad_output *= (1.0 - reward * 0.3)

        # 反向传播
        # W2 梯度
        grad_W2 = np.outer(grad_output, self._last_hidden)
        grad_b2 = grad_output

        # 隐藏层梯度
        grad_hidden = self.W2.T @ grad_output
        grad_hidden *= (self._last_hidden > 0)  # ReLU 导数

        # W1 梯度
        grad_W1 = np.outer(grad_hidden, self._last_input)
        grad_b1 = grad_hidden

        # 更新权重
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        # 权重裁剪
        self.W1 = np.clip(self.W1, -1.0, 1.0)
        self.W2 = np.clip(self.W2, -1.0, 1.0)

        # 学习率衰减
        self.lr *= self.lr_decay
        self.lr = max(0.0001, self.lr)

        # 返回损失
        loss = -np.log(probs[target_idx] + 1e-10)
        return float(loss)

    def train_sequence(self, neural_activity, target_text, reward=0.0):
        """训练一个完整序列"""
        self._reset_context()
        total_loss = 0.0

        for char in target_text:
            if char in self.char_to_idx:
                loss = self.train_step(neural_activity, char, reward)
                total_loss += loss

        return total_loss / max(1, len(target_text))

    def store_training_pair(self, neural_activity, target_text, reward):
        """存储训练对"""
        self.training_pairs.append({
            'neural': np.array(neural_activity, dtype=np.float32).copy(),
            'target': target_text,
            'reward': reward,
        })

    def replay_training(self, batch_size=10):
        """从缓冲区回放训练"""
        if len(self.training_pairs) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.training_pairs), batch_size,
                                    replace=False)
        total_loss = 0.0
        for idx in indices:
            pair = self.training_pairs[idx]
            loss = self.train_sequence(
                pair['neural'], pair['target'], pair['reward'])
            total_loss += loss

        return total_loss / batch_size

    def _softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-10)

    def get_stats(self):
        """获取统计信息"""
        return {
            'vocab_size': self.vocab_size,
            'lr': self.lr,
            'training_pairs': len(self.training_pairs),
            'W1_norm': float(np.linalg.norm(self.W1)),
            'W2_norm': float(np.linalg.norm(self.W2)),
        }
