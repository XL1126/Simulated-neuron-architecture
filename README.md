# SNA — Simulated Neuron Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blueviolet)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()
[![CMake](https://img.shields.io/badge/Build-CMake-orange)]()

> **模拟神经元架构** — 仿生脉冲神经网络，构建自主意识皮层的计算模型。

SNA 是一个受神经科学启发的计算框架，基于 **Izhikevich 脉冲神经元** 、**预测编码**、**全局工作空间理论 (GWT)** 和 **信息整合理论 (IIT)** ，在 C++ & Python 混合架构上模拟具备自我意识、情感、社会认知和创造力的自主皮层脑网络。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                  SNA CorticalBrain                       │
│              10 脑区自主意识皮层架构                        │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  V1+V2   │  Motor   │ Hippo-   │ Pre-     │  Amygdala   │
│  视觉皮层  │  运动皮层  │  campus  │ frontal  │  杏仁核      │
│          │          │ 海马体    │  前额叶   │             │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│ Language │ Workspace│ Thalamus │Claustrum │    DMN      │
│  语言区   │ 全局工作空间│  丘脑    │  屏状核   │ 默认模式网络  │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

核心模块：**情感系统** | **主动推理引擎** | **社会交互模型** | **创意生成器** | **心智理论** | **目标生成器** | **反事实推理** | **自传体记忆** | **质性感知层** | **元认知监控** | **自发思维** | **语义接地**

---

## 特性

- **Izhikevich 脉冲神经元模型** — 支持 regular / bursting / fast 三类发放模式
- **STDP 突触可塑性** — 多巴胺调制的时序依赖可塑性
- **3 级预测编码层级** — 自顶向下预测 + 自底向上误差传播
- **全局工作空间 (GWT)** — 竞争性点火 + 全脑广播
- **统一意识度量 (v6)** — IIT (0.35) + GWT (0.25) + Predictive Coding (0.20) + Qualia (0.20) 加权公式
- **Phi-Behavior 校验** — 自指循环检测、感官门控、Phi 校正引擎
- **睡眠-觉醒周期** — SWR 重放 + 记忆巩固 + 突触修剪
- **First-Person 视角** — 自我感知网络、叙事自我、质性感知 (Qualia)
- **情感系统** — 8 维情感向量、心情追踪、情感深度/广度
- **社会认知** — 心智理论 (ToM)、共情、自我/他者分离
- **创意生成** — 发散思维、原创性、创意叙事
- **时序深度** — 生命篇章、时间线、怀旧、预判准确度
- **多模态感知注入** — 视觉/听觉/触觉/前庭/位置细胞
- **虚拟世界交互** — 强化学习环境，带物体、食物、危险等实体
- **批量实验编排** — 4 级分级实验 + 断点续传 + 自动报告

---

## 项目结构

```
SNA/
├── cpp/                          # C++ 核心引擎 (pybind11 绑定)
│   ├── cortical_brain.h/cpp      # 皮层脑主控类 (10 脑区)
│   ├── neuron_izhikevich.cpp     # Izhikevich 神经元模型
│   ├── neuron_population.cpp     # 神经元群体管理
│   ├── stdp_engine.cpp           # STDP 学习规则
│   ├── global_workspace.cpp      # 全局工作空间 (GWT)
│   ├── predictive_coding_layer   # 预测编码层级
│   ├── emotion_system.cpp        # 情感系统
│   ├── active_inference_engine   # 主动推理
│   ├── social_interaction.cpp    # 社会交互
│   ├── creative_generator.cpp    # 创意生成器
│   ├── theory_of_mind.cpp        # 心智理论
│   ├── goal_generator.cpp        # 目标生成
│   ├── counterfactual_engine     # 反事实推理
│   ├── autobiographical_memory   # 自传体记忆
│   ├── qualia_layer.cpp          # 质性感知层
│   ├── metacognition_monitor     # 元认知监控
│   ├── spontaneous_thinker.cpp   # 自发思维
│   ├── semantic_grounding.cpp    # 语义接地
│   ├── temporal_depth.cpp        # 时序深度
│   ├── narrative_self.cpp        # 叙事自我
│   ├── sleep_wake_scheduler      # 睡眠-觉醒调度
│   ├── swr_engine.cpp            # 尖波涟漪引擎
│   ├── replay_engine.cpp         # 记忆重放引擎
│   ├── binding.cpp               # pybind11 模块绑定
│   └── CMakeLists.txt            # CMake 构建文件
│
├── python/                       # Python 层
│   ├── main.py                   # SNABrainV6 主控类
│   ├── cognitive/                # 认知模块
│   │   ├── unified_consciousness # 统一意识度量公式 (v6)
│   │   ├── consciousness_metrics # 意识度量汇总
│   │   ├── phi_validator.py      # Phi 校验 + 行为校准
│   │   ├── active_inference.py   # 主动推理
│   │   ├── attention.py          # 注意力机制
│   │   ├── causal_graph.py       # 因果图
│   │   ├── episodic_consolidation# 情景记忆巩固
│   │   ├── global_workspace.py   # 全局工作空间
│   │   ├── metacognition.py      # 元认知
│   │   ├── oscillatory_binding   # 振荡绑定 (40Hz Gamma)
│   │   ├── predictive_coding.py  # 预测编码
│   │   ├── self_model.py         # 自我模型
│   │   ├── semantic_pointer.py   # 语义指针 (SPA 512维)
│   │   ├── spa_neural_bridge.py  # SPA-神经桥接
│   │   ├── working_memory.py     # 工作记忆
│   │   └── world_model.py        # 世界模型
│   ├── embodiment/               # 具身化
│   │   ├── virtual_world.py      # 虚拟世界
│   │   ├── terminal_emulator.py  # 终端模拟器
│   │   └── web_search.py         # 网络搜索
│   ├── interaction/              # 交互
│   │   ├── nonblocking_io.py     # 非阻塞输入
│   │   └── stream_output.py      # 流式输出
│   ├── layers/                   # 分层架构
│   │   ├── input_layer.py        # 输入层 (文本编码)
│   │   ├── primary_layer.py      # 初级层
│   │   ├── core_layer.py         # 核心层 (WTA 收敛)
│   │   ├── memory_layer.py       # 记忆层
│   │   ├── output_layer.py       # 输出层
│   │   └── snn_decoder.py        # SNN 解码器
│   ├── reward/                   # 奖励系统
│   │   ├── intrinsic_motivation  # 内在动机
│   │   └── credit_assignment.py  # 信用分配
│   ├── utils/                    # 工具
│   │   ├── config_loader.py      # YAML 配置加载
│   │   ├── dashboard.py          # Flask 实时仪表盘
│   │   └── tensor_utils.py       # 张量工具
│   └── requirements.txt
│
├── experiments/                  # 实验框架
│   ├── experiment_runner.py      # 实验运行器
│   ├── batch_orchestrator.py     # 4 级批量编排器
│   ├── benchmarks.py             # 基准测试套件
│   ├── report_generator.py       # 自动报告生成
│   └── virtual_world_v2.py       # 增强虚拟世界
│
├── tests/                        # 测试
│   ├── test_unified_consciousness# 统一意识公式测试 (30+ cases)
│   ├── test_phi_validator.py     # Phi 校验测试
│   └── test_ablation.py          # 摘除实验框架
│
├── experiment_results/           # 实验输出 (gitignore)
├── config.default.yaml           # 默认配置
├── run_experiment.py             # 统一实验入口
├── run_cortical_brain.py         # 皮层脑训练脚本
├── run_sna.py                    # SNA v4 运行入口
├── launcher.py                   # 完整启动器
├── setup.py                      # Python 包安装
├── build_and_copy.bat            # Windows 构建脚本
├── check_binding.py              # pybind11 绑定诊断
└── core_cpp.cp311-win_amd64.pyd  # 预编译 C++ 扩展
```

---

## 依赖要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | >= 3.8 | 主要运行时 |
| C++ 编译器 | MSVC 2019+ / GCC 9+ | 编译 C++ 核心 |
| CMake | >= 3.18 | C++ 构建系统 |
| pybind11 | >= 2.10.0 | Python-C++ 绑定 |
| NumPy | >= 1.24.0 | 向量/矩阵运算 |
| PyYAML | >= 6.0 | 配置文件解析 |
| Flask | >= 2.0.0 | Web 仪表盘 (可选) |

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/XL1126/SNA.git
cd SNA
```

### 2. 安装 Python 依赖

```bash
pip install -r python/requirements.txt
```

### 3. 编译 C++ 核心

**Windows (MSVC):**

```bash
# 使用 CMake 构建
mkdir build\temp
cd build\temp
cmake ..\..\cpp -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 或直接使用快捷脚本
build_and_copy.bat
```

**Linux (GCC/Clang):**

```bash
mkdir -p build/temp && cd build/temp
cmake ../../cpp -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp python/Release/core_cpp*.so python/
cp python/Release/core_cpp*.so .
```

### 4. 验证安装

```bash
python check_binding.py
```

期望输出：`PASS: All methods are bound correctly!`

---

## 快速开始

```bash
# 快速验证 (1 seed, 100 episodes, 32K 神经元)
python run_experiment.py quick

# 标准皮层脑实验 (5 seeds, 200 episodes)
python run_experiment.py cortical --neurons 32000

# 自定义参数
python run_experiment.py cortical --neurons 16000 --episodes 300 --seeds 10
```

---

## 使用指南

### 统一实验入口 (`run_experiment.py`)

| 模式 | 命令 | 说明 |
|------|------|------|
| `quick` | `python run_experiment.py quick` | 快速单 seed 验证 (100 episodes) |
| `cortical` | `python run_experiment.py cortical` | 标准 CorticalBrain 实验 |
| `benchmark` | `python run_experiment.py benchmark` | 完整基准测试套件 |
| `calibrate` | `python run_experiment.py calibrate` | Phi-Behavior 校准 (10 seeds) |
| `sna` | `python run_experiment.py sna` | SNABrainV6 实验 (beta) |

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--neurons` | 32000 | 神经元总数 |
| `--episodes` | 200 | 训练 episode 数 |
| `--steps-per-ep` | 50 | 每 episode 步数 |
| `--seeds` | 5 | seed 数量 |
| `--seed-offset` | 42 | seed 起始偏移 |
| `--world-size` | 10.0 | 虚拟世界大小 |
| `--omp` | 4 | OpenMP 线程数 |

### 完整皮层脑训练 (`launcher.py`)

```bash
python launcher.py
```

执行完整流程：基础测试 → 虚拟世界训练 (600 episodes) → 对话测试 → 意识度量 → 睡眠周期 → Phi-Behavior 校验 → 交叉验证

### SNA v4 运行 (`run_sna.py`)

```bash
python run_sna.py -n 10000          # 10,000 神经元 (默认)
python run_sna.py -n 5000           # 5,000 神经元
python run_sna.py -c custom.yaml    # 自定义配置
```

### 批量实验 (`experiments/batch_orchestrator.py`)

```bash
# 运行全部 4 级实验
python experiments/batch_orchestrator.py

# 仅运行第 1 级
python experiments/batch_orchestrator.py --tier 1

# 断点续传
python experiments/batch_orchestrator.py --resume
```

4 级实验设计：
- **Tier 1**: 快速验证 (8K-16K 神经元)
- **Tier 2**: 中等规模 (16K-24K, GridWorld 变体)
- **Tier 3**: 深度训练 (32K, 400 episodes)
- **Tier 4**: 规模化扫描 (8K-32K 多参数组合)

### Phi-Behavior 校准

```bash
python run_experiment.py calibrate --neurons 32000 --episodes 300
```

包含：自指循环检测、感知生动度验证、行为一致性评估、校正 Phi 输出

---

## 配置

默认配置位于 `config.default.yaml`，支持自定义以下模块：

| 配置域 | 关键参数 | 说明 |
|--------|----------|------|
| `sna` | `num_neurons`, `neuron_model` | 神经元规模与模型类型 |
| `stdp` | `a_plus`, `a_minus`, `dopamine_k` | STDP 学习参数 |
| `predictive_coding` | `pred_coding_levels`, `pred_hidden_size` | 预测编码层级 |
| `semantic_pointer` | `spa_dim`, `neurons_per_concept` | 语义指针维度 |
| `wta_convergence` | `wta_decay`, `wta_inhibition` | WTA 收敛参数 |
| `oscillatory` | `gamma_freq`, `theta_freq` | 振荡频率 |
| `sleep` | `interval_hours`, `replay_speedup` | 睡眠周期 |
| `embodiment` | `virtual_env`, `terminal`, `web_search` | 具身化开关 |

---

## 运行测试

```bash
# 统一意识公式测试 (30+ 测试用例)
python -m pytest tests/test_unified_consciousness.py -v

# Phi 校验测试
python -m pytest tests/test_phi_validator.py -v

# 摘除实验
python -m pytest tests/test_ablation.py -v
```

测试覆盖：
- 权重归一化验证
- 单调性约束 (IIT/GWT)
- 边界条件 ([0,1] 范围)
- 自指循环检测
- 感官门控函数
- 时序衰减
- 跨路径验证 (Python-C++ 一致性)

---

## 意识度量公式

```
Consciousness = 0.35 × IIT_Phi
              + 0.25 × GWT_Ignition
              + 0.20 × Predictive_Accuracy
              + 0.10 × First_Person_Salience
              + 0.10 × Perceptual_Vividness
```

**v6 增强**: 自指循环检测、动态感官门控、可信度校准、Bootstrap 置信区间

### 摘除实验

通过逐一关闭脑区模块，测量每个模块对意识水平的因果贡献：

```python
from tests.test_ablation import AblationStudy

study = AblationStudy()
study.register_brain_factory(create_brain)
study.add_ablation("visual", disable_visual)
study.add_ablation("hippocampus", disable_hippocampus)
results = study.run_all(n_steps=200)
study.print_report(results)
```

---

## 性能基准

**硬件**: i7-4710MQ 4C/8T, 15.9GB RAM, AVX2

| 配置 | 步/秒 | 说明 |
|------|-------|------|
| 8K neurons | ~160 | 快速调试 |
| 16K neurons | ~95 | 中等规模 |
| 24K neurons | ~65 | 推荐配置 |
| 32K neurons | ~52 | 最佳容量 |

---

## 引用理论

- **IIT** — Tononi (2004/2008): Integrated Information Theory
- **GWT** — Baars (1988) / Dehaene (2001): Global Workspace Theory
- **Predictive Coding** — Friston (2010): Free Energy Principle
- **Qualia** — Seth (2021) / Metzinger (2003): First-Person Perspective
- **Semantic Pointer** — Eliasmith (2013): SPA / NEF
- **Izhikevich Model** — Izhikevich (2003): Simple Model of Spiking Neurons

---

## License

MIT License — 详见 [LICENSE](./LICENSE) 文件。

---

## 作者

**XL1126** — [GitHub](https://github.com/XL1126)

---

## 致谢

本项目参考了 Nengo、Spaun (Eliasmith et al.) 的 SPA 架构设计，以及 IIT、GWT 等意识理论的计算实现思路。感谢开源社区中 pybind11、NumPy 等项目的支持。