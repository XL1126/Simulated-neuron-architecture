# SNA · 模拟神经元架构 (Simulated Neuron Architecture)

> 🧠 **持续活跃的类脑脉冲神经网络 · 自主思考 · 流式输出 · 迈向真正意识**
>
> 完全抛弃 Transformer，模拟人脑神经元与突触的生理机制，实现纯文本对话的 AI 架构。

---

## 📖 目录

- [一、项目简介](#一项目简介)
- [二、快速开始（5分钟上手）](#二快速开始5分钟上手)
- [三、运行方式详解](#三运行方式详解)
- [四、如何与 SNA 对话](#四如何与-sna-对话)
- [五、配置说明](#五配置说明)
- [六、项目结构](#六项目结构)
- [七、核心原理](#七核心原理)
- [八、常见问题](#八常见问题)

---

## 一、项目简介

### 这是什么？

SNA 是一个**完全不同于 ChatGPT/Transformer 的 AI 对话系统**。它不依赖任何传统的深度学习技术（没有矩阵乘法、没有反向传播、没有注意力机制），而是模拟人脑中约 1 万个神经元和 200 万个突触的工作方式。

### 核心理念

| 对比项 | ChatGPT / Transformer | SNA |
|--------|----------------------|-----|
| 运算方式 | 矩阵乘法 + 梯度下降 | 脉冲发放 + 突触可塑性 |
| 学习方式 | 反向传播训练 | STDP（脉冲时序依赖可塑性）在线学习 |
| 知识存储 | 固定权重 | 动态突触权重，持续演化 |
| 运行模式 | 请求-响应，无状态 | 持续活跃，永不停止思考 |
| 遗忘机制 | 无 | 自然遗忘 + 记忆巩固 |
| 回复方式 | 概率采样 (top-k, temperature) | 脉冲收敛 → 字符解码 |
| 硬件依赖 | GPU | CPU 即可（开发机） |

### ⚠️ 重要理念：非干扰原则

**开发者不教模型如何回复。** 代码只提供运行环境（神经元、突触、学习规则），模型如何联想、推理、回复**完全由突触权重在线演化决定**。

代码中不存在：
- ❌ 任何 `if 用户说"你好" → 回复"你好"` 这种硬编码
- ❌ 任何预设的对话模板、角色设定
- ❌ 任何回复字数限制
- ❌ 任何概率采样策略

---

## 二、快速开始（5分钟上手）

### 前置条件

你的电脑需要装好 Python（你的环境已满足 ✅）：

| 工具 | 状态 |
|------|------|
| Python 3.11.0 | ✅ |
| CMake 4.2.3 | ✅ |
| MSVC (Visual Studio 2022) | ✅ |

### 启动步骤

#### 第 1 步：进入项目目录

打开终端（PowerShell 或 CMD），输入：

```powershell
cd d:\SNA架构\SNA
```

#### 第 2 步：确认 C++ 核心已编译

输入以下命令验证：

```powershell
python -c "import core_cpp; print('C++核心加载成功！神经元数量:', core_cpp.NeuronPopulation(100,10).size())"
```

如果看到 `C++核心加载成功！神经元数量: 100`，说明一切正常。

> 如果这一步报错，说明 C++ 扩展需要重新编译。请跳到 [常见问题](#八常见问题) 查看编译方法。

#### 第 3 步：启动 SNA

**最简单的方式（1000 个神经元，适合快速体验）：**

```powershell
python run_sna.py
```

启动后你会看到：

```
============================================================
  SNA - Simulated Neuron Architecture
  Neurons: 1000  |  Mode: Console
  Non-interference Principle: NO hardcoded replies, NO templates
============================================================
[SNA] Initializing with 1000 neurons, avg_degree=50
[SNA] C++ Core available: True
[SNA] Starting continuous active loop...
[SNA] Type your message and press Enter. Type 'quit' to exit.
[SNA] Type '+' to like, '-' to dislike the response.
--------------------------------------------------
```

#### 第 4 步：开始对话

在提示符后面直接输入你想说的话，按回车发送：

```
你好
```

SNA 会在思考一段时间后流式输出回复（一个字一个字地输出，模拟人类边想边说的过程）。

#### 第 5 步：给反馈

- 输入 `+` 按回车 → 表示你喜欢刚才的回复（增加多巴胺奖励）
- 输入 `-` 按回车 → 表示你不喜欢刚才的回复（减少多巴胺惩罚）

#### 第 6 步：退出

输入 `quit` 或 `exit` 按回车退出。

---

## 三、运行方式详解

### 方式一：run_sna.py（推荐，最简单）

```powershell
# 默认 1000 个神经元
python run_sna.py

# 指定神经元数量
python run_sna.py -n 10000    # 1万个（标准配置）
python run_sna.py -n 100      # 100个（极简模式）
python run_sna.py -n 100000   # 10万个（需要大内存）

# 开启网页仪表盘
python run_sna.py -n 500 -d

# 使用自定义配置文件
python run_sna.py -c config.default.yaml
```

### 方式二：python main.py（完整控制）

```powershell
cd python
python main.py --neurons 5000 --dashboard
```

### 各种规模的建议

| 规模 | 神经元数 | 内存占用 | 适用场景 |
|------|---------|---------|---------|
| 极简 | 100 | ~100MB | 快速测试，验证代码能跑 |
| 体验 | 1000 | ~300MB | **新手推荐**，有基本对话能力 |
| 标准 | 10000 | ~2GB | 完整版，较好的对话体验 |
| 扩展 | 100000 | ~20GB | 需要 32GB+ 内存的机器 |

---

## 四、如何与 SNA 对话

### 基本对话

SNA 是一个**刚出生的大脑**——初始时所有突触权重都是随机的，它**什么都不知道**。你需要像教一个婴儿一样慢慢和它交流。

```
你：你好
SNA：（可能输出一些随机字符或简单词汇）
你：+
SNA：（收到奖励，相关突触被加强）

你：我叫小明
SNA：（可能输出一些内容）
你：+

你：我叫什么？
SNA：（尝试回忆，可能输出与"小明"相关的内容）
```

### 给 SNA 反馈很重要

SNA 通过你的反馈来学习：

- 输入 `+`：告诉 SNA "这个回答不错" → 多巴胺 +0.2，相关突触加强
- 输入 `-`：告诉 SNA "这个回答不好" → 多巴胺 -0.2，相关突触减弱

**建议**：每次 SNA 给出回复后，都给一个 `+` 或 `-`，帮助它快速学习。

### 对话技巧

1. **耐心**：刚启动时 SNA 的输出会很随机，这是正常的。就像婴儿不会一出生就会说话。
2. **重复**：多次告诉 SNA 同一个事实（如你的名字），它会慢慢记住。
3. **简短**：开始时用简短的句子，逐步增加复杂度。
4. **反馈**：好的回复给 `+`，不好的给 `-`，这是 SNA 唯一的学习信号。
5. **不打断**：SNA 在持续思考，突然切换话题可能导致输出混乱。

### 可以用 SNA 做什么？

- 💬 **基本对话**：和它聊天
- 🔧 **终端操作**：SNA 可以学会执行命令行（如 `ls`、`pwd`、`echo`）
- 🌐 **联网搜索**：配置开启后，SNA 可以学会搜索信息
- 🎮 **虚拟世界**：配置开启后，SNA 可以在虚拟世界中探索（look, move, take, talk）

---

## 五、配置说明

配置文件是 `config.default.yaml`，你可以用记事本打开修改。

### 核心配置项

```yaml
sna:
  num_neurons: 10000    # 神经元数量，越大对话能力越强，但越慢
  avg_degree: 200       # 每个神经元的平均连接数
  dendrite_compartments: 2  # 树突隔室数

stdp:
  a_plus: 0.01          # STDP 增强系数（越大学习越快，但可能不稳定）
  a_minus: 0.012        # STDP 减弱系数
  dopamine_k: 1.0       # 多巴胺影响强度

output:
  streaming_window_ms: 50   # 流式输出窗口大小
  similarity_threshold: 0.7 # 字符识别阈值

embodiment:
  virtual_env: true     # 是否开启虚拟世界（需要更多内存）
  terminal: true        # 是否允许终端操作（安全模式下会阻止危险命令）
  web_search: true      # 是否允许联网搜索

sleep:
  interval_hours: 8     # 每隔多少小时进入睡眠整理
  duration_hours: 1     # 每次睡眠持续多久
```

### 快速调参建议

| 想要的改变 | 修改项 |
|-----------|--------|
| 让 SNA 更聪明 | 增加 `num_neurons` 和 `avg_degree` |
| 让 SNA 学得更快 | 增加 `a_plus`（建议不要超过 0.02） |
| 让 SNA 更健忘 | 增加 `forgetting.rate_per_10000steps` |
| 关闭虚拟世界节省内存 | `embodiment.virtual_env: false` |

---

## 六、项目结构

```
SNA/
├── cpp/                              # 🔧 C++ 高性能核心
│   ├── neuron_izhikevich.h/cpp       #    神经元模型（Izhikevich方程）
│   ├── dendrite_compartment.h/cpp    #    树突隔室计算
│   ├── synapse.h                     #    突触数据结构
│   ├── spike_event.h                 #    脉冲事件定义
│   ├── stdp_engine.h/cpp             #    STDP学习规则
│   ├── neuron_population.h/cpp       #    神经元群体管理
│   ├── binding.cpp                   #    pybind11 Python绑定
│   └── CMakeLists.txt                #    CMake构建文件
│
├── python/                           # 🐍 Python 高层逻辑
│   ├── layers/                       #    五层架构
│   │   ├── input_layer.py            #      输入层：文字→脉冲
│   │   ├── primary_layer.py          #      初级层：语义分组
│   │   ├── core_layer.py             #      核心层：联想推理
│   │   ├── memory_layer.py           #      记忆层：三层存储+遗忘
│   │   └── output_layer.py           #      输出层：脉冲→文字（流式）
│   ├── cognitive/                    #    认知模块
│   │   ├── working_memory.py         #      工作记忆（7个槽位）
│   │   ├── attention.py              #      注意力机制
│   │   ├── world_model.py            #      世界预测模型
│   │   └── metacognition.py          #      元认知监控
│   ├── reward/                       #    奖励系统
│   │   ├── intrinsic_motivation.py   #      内在动机（好奇/新颖性）
│   │   └── credit_assignment.py      #      长期信用分配
│   ├── embodiment/                   #    具身交互
│   │   ├── virtual_world.py          #      虚拟世界
│   │   ├── terminal_emulator.py      #      终端模拟器
│   │   ├── web_search.py             #      联网搜索
│   │   └── embodiment_interface.py   #      具身接口
│   ├── interaction/                  #    用户交互
│   │   ├── nonblocking_io.py         #      非阻塞输入
│   │   └── stream_output.py          #      流式输出
│   ├── utils/                        #    工具
│   │   ├── config_loader.py          #      配置加载
│   │   └── dashboard.py              #      Web仪表盘
│   └── main.py                       #    主循环入口
│
├── config.default.yaml               # 📋 默认配置文件（用记事本打开）
├── run_sna.py                        # 🚀 一键启动脚本
├── setup.py                          # 📦 pip安装支持
└── README.md                         # 📖 本说明文档
```

---

## 七、核心原理

### 神经元如何工作？

每个神经元就像一个小小的"电容器"：
1. 接收来自其他神经元的脉冲信号（电信号输入）
2. 膜电位逐渐升高（充电）
3. 超过阈值时发放脉冲（放电）
4. 脉冲通过突触传递给下游神经元

### 突触如何学习？

SNA 使用 **STDP（脉冲时序依赖可塑性）** 学习：
- 如果 A 神经元在 B 神经元**之前**放电 → 连接加强（"A 导致了 B"）
- 如果 A 神经元在 B 神经元**之后**放电 → 连接减弱（"A 和 B 无关"）

这就是"**一起放电的神经元会连接在一起**"——和真实大脑一样。

### 多巴胺做什么？

多巴胺就像"奖励信号"：
- 多巴胺高 → STDP 学习效率高 → 当前活跃的突触被加强
- 多巴胺低 → STDP 学习效率低 → 当前活跃的突触被减弱

你输入 `+` 或 `-` 就是在控制多巴胺水平。

### 为什么要"睡眠"？

SNA 每模拟运行 8 小时会进入睡眠阶段（模拟 1 小时），期间：
1. 删除最弱的突触（"遗忘不重要的东西"）
2. 巩固重要的突触（"记住重要的东西"）
3. 回放之前的记忆（"梦到白天发生的事"）

---

## 八、常见问题

### Q1：启动后 SNA 输出的全是乱码？

**A1**：这是完全正常的。SNA 刚启动时，所有突触权重都是随机的，就像一个刚出生的婴儿。你需要：
1. 耐心地和它对话
2. 对好的回复输入 `+`
3. 多次重复相同的内容帮助它学习

### Q2：SNA 没有任何反应？

**A2**：请检查：
```powershell
python -c "import core_cpp; print('OK')"
```
如果没有输出 `OK`，说明 C++ 核心没有编译成功。请重新编译：

```powershell
# 只用CMake编译（最简单）
cd d:\SNA架构\SNA
mkdir build -Force | Out-Null
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="C:\Users\xiaoli\AppData\Local\Programs\Python\Python311\Lib\site-packages\pybind11\share\cmake\pybind11"
cmake --build build --config Release
Copy-Item "build\Release\core_cpp.cp311-win_amd64.pyd" -Destination "." -Force
```

### Q3：如何让 SNA 更聪明？

**A3**：增加神经元数量：
```powershell
python run_sna.py -n 10000
```
更多的神经元 = 更强的表达能力。但也会占用更多内存和 CPU。

### Q4：SNA 会记住我们的对话吗？

**A4**：会，但不完美。SNA 有三层记忆：
- **瞬时记忆**：最近 10 秒的对话
- **短期记忆**：最近几小时的对话，会逐渐遗忘
- **长期记忆**：反复出现的模式会被固化

重启程序后，所有记忆会丢失（当前版本不支持保存到磁盘）。

### Q5：能同时开多个 SNA 吗？

**A5**：可以，每个终端窗口运行一个独立的 SNA 实例。但它们之间不会互相通信。

### Q6：能用 GPU 加速吗？

**A6**：当前版本使用纯 CPU（开发机完全够用）。1 万个神经元在 Intel i7 上跑得很流畅。GPU 支持是未来的可选功能。

### Q7：命令行参数完整列表？

**A7**：
```powershell
python run_sna.py --help
```

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--neurons` | `-n` | 神经元数量 | 1000 |
| `--config` | `-c` | 配置文件路径 | 自动生成 |
| `--dashboard` | `-d` | 开启 Web 仪表盘 | 关闭 |
| `--dashboard-port` | `-p` | 仪表盘端口 | 5050 |

### Q8：回复速度太慢怎么办？

**A8**：
1. 减少神经元数量：`python run_sna.py -n 500`
2. 关闭虚拟世界：修改配置 `embodiment.virtual_env: false`
3. 关闭搜索功能：修改配置 `embodiment.web_search: false`

### Q9：SNA 输出太长/太短怎么办？

**A9**：SNA 不限制输出长度。回复长度由内部脉冲收敛过程自然决定。如果想让它多说点，可以：
- 多给 `+` 反馈鼓励它
- 用更开放的问题引导它

### Q10：如何查看后台运行状态？

**A10**：开启 Dashboard 模式：
```powershell
python run_sna.py -d
```
然后在浏览器打开 `http://127.0.0.1:5050`，可以看到实时运行状态。

---

## 📝 给开发者的提醒

本项目严格遵守**非干扰原则**：

- ✅ **允许的**：提供神经元动力学、STDP学习规则、脉冲编解码、奖励信号机制
- ❌ **禁止的**：硬编码回复规则、对话模板、字数限制、概率采样策略

模型的想法和行为完全由突触权重的在线演化决定。开发者只管"物理定律"，不管"意识形态"。

---

> **"我们不是在造一台会说话的机器，而是在种一颗会思考的种子。"**
>
> —— SNA 设计哲学

---

*最后更新：2026-05-07*
