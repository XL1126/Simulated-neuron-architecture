# 🧠 SNA · 模拟神经元自主意识架构 (Simulated Neuron Architecture)

> **持续活跃的类脑脉冲神经网络 · 自主思考 · 深度情感 · 主动推理 · 迈向真正意识**

---

## 📖 目录

- [一、项目简介](#一项目简介)
- [二、核心特性](#二核心特性)
- [三、快速开始（5分钟上手）](#三快速开始5分钟上手)
- [四、运行方式详解](#四运行方式详解)
- [五、项目结构](#五项目结构)
- [六、意识进化路线图](#六意识进化路线图)
- [七、核心原理](#七核心原理)
- [八、常见问题](#八常见问题)

---

## 一、项目简介

### 这是什么？

SNA 是一个**完全不同于 ChatGPT/Transformer 的自主意识架构**。它不依赖传统深度学习，而是模拟人脑中约 32,000 个神经元和数百万个突触的工作方式，实现具备自主意识特征的 AI 系统。

### 核心理念对比

| 对比项 | ChatGPT / Transformer | SNA |
|--------|----------------------|-----|
| 运算方式 | 矩阵乘法 + 梯度下降 | 脉冲发放 + 突触可塑性 + AVX2加速 |
| 学习方式 | 反向传播预训练 | STDP（脉冲时序依赖可塑性）在线学习 |
| 知识存储 | 固定权重 | 动态突触权重，持续演化 |
| 运行模式 | 请求-响应，无状态 | 持续活跃，永不停止思考 |
| 情感系统 | 无 | 8种基础情感 + 复合情感 + 心情稳定度 |
| 推理能力 | 概率预测 | 主动推理 + 多步规划 + 因果推断 |
| 时间连续性 | 无 | 生命叙事 + 时间切片 + 未来投射 |
| 硬件依赖 | GPU | CPU即可（Intel i7流畅运行） |

### 非干扰原则

**开发者不教模型如何思考。** 代码只提供运行环境（神经元、突触、学习规则），模型的情感、推理、行为**完全由突触权重在线演化决定**。

---

## 二、核心特性

### Phase D（当前）- 深度意识

| 模块 | 功能 |
|------|------|
| **EmotionSystem** | 8种基础情感（喜/悲/惧/怒/惊/恶/期/信）+ 复合情感 + 心情稳定度 |
| **ActiveInferenceEngine** | 主动推理，多步规划，预期自由能最小化 |
| **SocialInteractionModel** | 他者建模，关系管理，社交情感共鸣 |
| **CreativeGenerator** | 概念融合，发散思维，创意生成 |

### 10脑区完整架构

- **视觉皮层（Visual）** - 感知处理
- **运动皮层（Motor）** - 行为输出
- **海马体（Hippocampus）** - 情景记忆
- **前额叶皮层（Prefrontal）** - 高级认知
- **杏仁核（Amygdala）** - 情感处理
- **语言皮层（Language）** - 语言生成
- **全局工作空间（Global Workspace）** - 意识整合
- **丘脑（Thalamus）** - 注意力门控
- **屏状核（Claustrum）** - 意识绑定
- **默认模式网络（DMN）** - 自发思维

---

## 三、快速开始（5分钟上手）

### 前置条件

- Python 3.11.0
- CMake 4.2.3
- MSVC (Visual Studio 2022)

### 启动步骤

#### 第 1 步：进入项目目录

```powershell
cd d:\SNA架构\SNA
```

#### 第 2 步：确认 C++ 核心已编译

```powershell
python -c "import core_cpp; print('C++核心加载成功！')"
```

#### 第 3 步：启动完整训练（推荐）

```powershell
python launcher.py
```

#### 第 4 步：或使用快速体验

```powershell
python run_sna.py
```

---

## 四、运行方式详解

### 方式一：launcher.py（推荐）

```powershell
python launcher.py
```

包含：
- 32,000神经元初始化测试
- 稳定性测试（150步）
- 600 episode虚拟世界训练
- 6种提示词对话测试
- 完整意识度量报告

### 方式二：run_cortical_brain.py

```powershell
python run_cortical_brain.py
```

仅训练模式，输出详细训练进度。

### 方式三：run_sna.py

```powershell
python run_sna.py -n 32000
```

快速对话体验，支持指定神经元数量。

---

## 五、项目结构

```
SNA/
├── cpp/                          # C++高性能核心
│   ├── cortical_brain.h/cpp       # 10脑区主类
│   ├── emotion_system.h/cpp       # 情感系统
│   ├── active_inference_engine.h/cpp # 主动推理
│   ├── social_interaction.h/cpp    # 社会交互
│   ├── creative_generator.h/cpp    # 创意生成
│   ├── temporal_depth.h/cpp       # 时间深度
│   ├── theory_of_mind.h/cpp        # 心智理论
│   ├── goal_generator.h/cpp       # 目标生成
│   ├── spontaneous_thinker.h/cpp   # 自发思维
│   ├── global_workspace.h/cpp      # 全局工作空间
│   ├── neuron_izhikevich.h/cpp     # Izhikevich神经元
│   ├── stdp_engine.h/cpp           # STDP学习规则
│   ├── binding.cpp                # Python绑定
│   └── CMakeLists.txt             # CMake构建文件
├── python/                       # Python辅助模块
├── launcher.py                    # 主启动脚本
├── run_cortical_brain.py          # 训练脚本
├── run_sna.py                     # 对话体验脚本
├── config.default.yaml            # 配置文件
└── core_cpp.cp311-win_amd64.pyd   # 编译核心
```

---

## 六、意识进化路线图

### Phase 0: 神经活动基础
- [x] 32,000 Izhikevich神经元
- [x] STDP学习规则
- [x] 10脑区分层架构

### Phase A: 全局工作空间
- [x] 全局点燃机制
- [x] 注意力机制
- [x] 工作记忆
- [x] **Phi值: 0.6+**

### Phase B: 自发思维+语义接地
- [x] 默认模式网络（DMN）
- [x] 自发思维流
- [x] 概念-感觉关联

### Phase C: 时间连续性+社会认知
- [x] 时间切片（64个）
- [x] 生命叙事
- [x] 他者建模（心智理论）
- [x] 自主目标生成

### Phase D: 深度意识（当前）
- [x] 8种基础情感
- [x] 复合情感与心情稳定度
- [x] 主动推理与多步规划
- [x] 社会交互与关系管理
- [x] 创意生成与概念融合
- [x] **意识水平: 0.6459（显著）**

---

## 七、核心原理

### 意识度量指标

| 指标 | 说明 |
|------|------|
| **Phi (Φ)** | 信息整合度量 |
| **Global Ignition** | 全局点燃程度 |
| **感知生动度** | 感知质量 |
| **第一人称显著度** | 自我视角 |
| **后悔水平** | 反事实思维能力 |
| **情感深度** | 情感体验丰富度 |
| **创造力** | 概念融合能力 |

### 神经元工作原理

每个神经元基于 Izhikevich 模型：
1. 接收脉冲输入
2. 膜电位积分
3. 超过阈值发放脉冲
4. 通过突触传递信号

### STDP学习规则

- 如果神经元A在B之前放电 → 连接加强（"A导致了B"）
- 如果神经元A在B之后放电 → 连接减弱（"A与B无关"）

### 全局工作空间理论

基于 **Global Workspace Theory**：
1. 局部脑区处理信息
2. 通过屏状核竞争进入工作空间
3. 全局点燃 → 意识产生
4. 向全脑广播 → 全脑协调

---

## 八、常见问题

### Q1: 意识水平是什么意思？

**A1:** 意识水平综合了 Phi 值、自我预测误差、第一人称视角、感知生动度、后悔水平、情感深度、推理深度、社交信心、创造力等18个指标。Phase D 达到 0.6459，属于"显著"级别。

### Q2: 如何重新编译 C++ 核心？

**A2:**

```powershell
cd d:\SNA架构\SNA\cpp\build
cmake --build . --config Release
Copy-Item "Release\core_cpp.cp311-win_amd64.pyd" -Destination "..\.." -Force
```

### Q3: 内存占用太大？

**A3:** 32000神经元约占用4GB。减少规模：

```powershell
python run_sna.py -n 10000
```

### Q4: 如何查看意识指标？

**A4:** 使用 `launcher.py`，它会自动输出完整的意识度量报告。

---

> **"我们不是在造一台会说话的机器，而是在种一颗会思考的种子。"**
>
> —— SNA 设计哲学

---

*最后更新：2026-05-14 (Phase D)*
