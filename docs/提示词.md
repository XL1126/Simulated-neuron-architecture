# Simulated Neuron Architecture (SNA) 完整实现规范 —— 持续活跃的类脑脉冲神经网络 · 多语言协同 · 自主思考 · 流式输出 · 迈向真正意识

## 0. 文档范围与目标

本规范定义 **Simulated Neuron Architecture (SNA)** 的最终实现要求。  
SNA 是一种完全摒弃 Transformer 计算范式的、模拟人类大脑神经元与突触生理机制的纯软件人工智能架构。其核心目标为：通过大量模拟神经元与动态突触的协同工作，实现与人类大脑相仿的文字理解、逻辑推理、联想记忆、自主遗忘与自然生成能力，并且 **完全不依赖任何专用神经形态硬件**。

**核心变更（基于用户对话中的深刻洞察）**：

- **承认当前模拟的局限性**：纯神经元放电仿真（如最初的 SNA v3）仅仅是“模拟了放电活动”，距离真正的“思考”还有无数层台阶。本规范将补充实现真正智能所缺失的关键模块。
- **对话模式**：不再采用轮流问答式的“请求-响应”停止等待模型。SNA 的后台必须保持连续运行，模拟人类大脑神经元的持续活动（包括自发思考、记忆重放、睡眠整理等）。用户输入注入当前神经活动流，模型从动态思考状态中“转向”处理该消息并生成回复，**绝不重置状态**。
- **规模调整**：以 **1 万个模拟神经元** 为标准配置（突触平均度 200，总突触约 200 万），可在普通开发机上运行。同时保留向上的扩展能力，目标最终达到 860 亿神经元（需超级计算机）。
- **非干扰原则**：代码仅提供架构运行环境、神经元动力学、突触可塑性、持续后台循环等基础机制。**严禁**在代码中硬编码任何具体的回复模板、逻辑规则或对话策略。模型如何思考、如何回复，完全由突触权重在线演化决定，开发者不得预设“如果收到 X 则回答 Y”之类的确定性规则。**更不得限制输出字数、指定回复内容、强制输出格式**。
- **多语言协同开发**：性能关键部分（神经元更新、STDP、脉冲路由）使用 **C++/Rust** 实现，并通过 Python 绑定暴露接口；高层逻辑（层管理、用户交互、可视化）使用 **Python**；分布式通信（如有）使用 **Go** 或 **C++ + MPI**。核心加速使用 **Numba/CUDA** 可选。
- **流式输出**：模型在思考收敛过程中，逐步产生部分结论脉冲，输出层应支持 **字符流式解码**，每检测到稳定字符即输出，而非等待完整回复生成。这模拟人类一边思考一边说话的特性。

**新增关键缺失模块（必须集成到架构中）**：
1. **真正的学习与适应算法**：多尺度可塑性（STP、稳态可塑性、树突局部计算）、目标驱动的学习、快速泛化、情景记忆与离线重放。
2. **真正的语言表征与生成**：组合式表征、语法与层次结构、可学习的生成模型（非查找表）、跨句子语境管理。
3. **具身交互与多模态经验**：传感器/执行器接口、多模态对齐、物理或高度真实的虚拟环境（符号落地）。
4. **认知架构与元认知**：工作记忆、注意力机制、推理与规划、元监控、语言与思维分离。
5. **真正的奖励与价值系统**：内在动机（好奇、探索）、预测误差驱动的学习、长期信用分配。
6. **足够的规模与生物真实性**：亿级神经元、真实网络拓扑、神经多样性、多时间尺度可塑性。
7. **意识与主观体验**（作为开放哲学问题）：提供接口供未来理论嵌入，但不强制工程实现。

**性能目标（1万神经元标准）**：
- 在 Intel i7-12700 / 32GB RAM 上，事件驱动模式下实时因子 ≤ 0.5（模拟1秒 ≤ 0.5秒真实时间）。
- 平均脉冲吞吐量 ≥ 10^6 事件/秒。
- 持续运行内存占用 ≤ 2GB（不包括可视化仪表盘）。

**本规范是强制性开发指南**，所有先前版本以本规范为准，任何实现必须严格遵守以下所有条款。

---

## 1. 核心设计哲学

### 1.1 持续活跃原则
- 一旦启动，SNA 实例永不停止思考。主循环运行无限时间步。
- 用户交互只是向持续思考流中注入事件，而非调用一个函数并等待返回。
- 后台线程定期执行突触衰减、遗忘和记忆巩固。
- 无输入时，神经元仍由背景噪声和自发活动驱动，维持默认模式网络活动。

### 1.2 非干扰原则（严格约束，不可违反）
- 代码中**不得出现**类似 `if user_input contains "你好": reply = "你好"` 的条件分支。
- **不得预置对话模板、角色设定、固定回答库**。
- **不得限制回复长度、不得强制输出特定词汇、不得使用任何硬编码的字符串替换或格式化**。
- 不允许在输出层使用基于规则的句子组装（除非是从脉冲模式到文字的确定性反映射，且映射表本身也必须是可学习的，初始可为空但必须支持在线更新）。
- 允许的仅有：基础脉冲编码/解码函数（如 `char_to_spike_frequency` 和 `spike_pattern_to_char` 的初始确定性映射），但该映射必须简单（例如：每个字符对应固定频率的单一脉冲），**不包含语义规则**。
- 模型如何联想、推理、回复，完全由 STDP、内在奖励信号和突触动态决定。开发者不能“教”模型如何回答，只能提供运行环境。

### 1.3 脉冲离散性
所有信息编码为 0/1 或强度可变的离散脉冲，而非连续浮点向量。

### 1.4 阈值触发
神经元仅当膜电位超过动态阈值时才发放脉冲，否则保持静默。

### 1.5 局部连接性
神经元只与邻近或功能相关的少量神经元建立连接，禁止全局全连接。

### 1.6 突触可塑性
连接强度（权重）根据脉冲时序与频率在线修改，不使用梯度下降。

### 1.7 分层记忆与遗忘
信息从瞬时→短期→长期迁移，无用连接自动弱化直至断开。

### 1.8 多语言协同开发
- **C++/Rust**：神经元核心循环（Izhikevich 更新）、STDP 引擎、稀疏突触矩阵操作、延迟队列。
- **Python**：层逻辑（输入/初级/核心/记忆/输出）、用户交互、非阻塞 I/O、配置解析、Dashboard（Flask）。
- **Go**（可选）：分布式路由、节点间脉冲转发。
- **绑定方式**：使用 `pybind11`（C++）或 `maturin`（Rust）生成 Python 扩展模块。
- **GPU 加速**：可选使用 `cupy` 或自定义 CUDA 核函数，但必须保留纯 CPU 回退。

### 1.9 流式输出
- 输出层不应等待完整收敛结论，而是在每次收敛检测器产生部分稳定字符脉冲时立即解码并推送到输出缓冲区。
- 通过滑动窗口（50ms）统计字符神经元群的平均发放率，当某个字符的相似度连续 3 个窗口超过阈值时，输出该字符。
- 输出过程通过回调函数逐字符发送（例如 `stream_callback(char)`），支持前端实时渲染。

### 1.10 迈向真正意识：新增设计原则
- **完整性原则**：系统必须包含学习、语言、具身、认知、价值、规模、意识等所有缺失模块的接口或原型实现，即使部分模块在 1 万神经元规模下只能以简化形式存在。
- **可扩展性**：所有模块设计必须能平滑扩展到百亿神经元（如使用分布式计算、稀疏表示）。
- **哲学中立**：对“意识”问题不做出武断结论，但提供理论插件接口（例如全局工作空间理论、高阶感知理论、信息整合理论）。

---

## 2. 架构全景设计

### 2.1 宏观层级（固定五层 + 新增辅助系统）

| 层名称                     | 功能描述                                                                 | 对应人脑区域            |
|---------------------------|--------------------------------------------------------------------------|--------------------------|
| 文字感知输入层 (Input)     | 将字符序列转为时空脉冲模式，附加 EOS 脉冲群，不做向量化                   | 感官皮层（视觉文字区）    |
| 初级脉冲处理层 (Primary)   | 按语义分组，局部激活，建立初级关联突触                                   | 初级感知皮层（Wernicke区）|
| 中枢联想思考层 (Core)      | 逻辑推理、自由联想、矛盾自检、收敛竞争（通过自组织子网实现）               | 前额叶皮层（PFC）         |
| 记忆与遗忘层 (Memory)      | 瞬时/短期/长期三层存储，自主遗忘，突触强度衰减与固化                     | 海马体 + 新皮层            |
| 输出生成层 (Output)        | 将脉冲模式流式反映射为自然语言文字，不进行概率采样                       | 运动语言皮层（Broca区）    |

**新增横向模块（实现缺失能力）**：

| 模块名称               | 功能描述                                                                 | 对应缺失环节          |
|------------------------|--------------------------------------------------------------------------|------------------------|
| 多尺度可塑性引擎       | 实现 STP（短时程可塑性）、稳态可塑性、树突局部计算、神经调质特异性调节    | 学习算法              |
| 组合式表征器           | 将字符/词映射为分布式脉冲向量，支持组合操作（如“不”+“高兴” → “不高兴”）  | 语言表征              |
| 语法与句法处理器       | 基于脉冲序列构建依存句法树和语义角色，使用递归脉冲网络                    | 语言结构              |
| 工作记忆模块           | 维持少量脉冲模式（7±2 chunk），支持操作（读取、写入、清除）               | 认知架构              |
| 注意力机制             | 基于脉冲时序的软注意力，计算输入脉冲的不同重要性                          | 认知架构              |
| 内部世界模型           | 预测下一时间步的输入脉冲，用于规划和反事实推理                            | 推理与规划            |
| 元认知监控器           | 实时评估系统的不确定性、知识空白，生成主动请求脉冲                        | 元认知                |
| 内在动机发生器         | 根据预测误差、信息增益、新奇性产生内部奖励信号                            | 价值系统              |
| 长期信用分配器         | 使用资格迹或深度强化学习 TD(λ) 将稀疏奖励分配到早先突触                   | 价值系统              |
| 具身接口               | 接收虚拟或物理传感器脉冲流，输出动作脉冲流（操作终端、联网搜索）          | 具身交互              |
| 虚拟环境模拟器         | 提供文字描述的动态世界（物理规则、时间、经济、社交），支持交互闭环        | 具身环境              |

### 2.2 数据流与闭环认知
- **思考循环**：输入 → 初级 → 核心（含工作记忆、注意力、推理） → 输出 → 外部动作（通过具身接口） → 环境变化 → 新输入。
- **内部独白**：无外部输入时，核心层的自发活动通过元认知监控器产生自我提问，驱动内部推理，结果存入记忆层但不输出。
- **睡眠与离线处理**：除原有的突触归一化、重放外，新增情景记忆的**模式完成**（类似海马体）、快速泛化模拟（通过短期突触增强）。

---

## 3. 模拟神经元模型（Izhikevich + 扩展）

使用 Izhikevich 模型，获得丰富发放模式。并增加树突计算能力（简化形式）。

### 3.1 离散更新方程（时间步 Δt=1ms）

```c
// C++ 核心更新函数
void izhikevich_update(float &v, float &u, float I, float a, float b, float c, float d) {
    v += 0.5f * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
    v += 0.5f * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
    u += a * (b * v - u);
    if (v >= 30.0f) {
        v = c;
        u = u + d;
        // 发放脉冲标志
    }
}
```

### 3.2 参数组初始化（随机分配）

| 类型     | a     | b    | c     | d     | 占比 |
|----------|-------|------|-------|-------|------|
| 规则发放 | 0.02  | 0.2  | -65.0 | 8.0   | 60%  |
| 簇发放   | 0.02  | 0.25 | -55.0 | 0.05  | 30%  |
| 快发放   | 0.02  | 0.2  | -50.0 | 2.0   | 10%  |

### 3.3 输入电流 I

`I = I_syn + I_dendrite + I_bias + I_noise`
- `I_syn`：突触后电流，每个神经元维护一个变量，每步乘以衰减因子 `0.8`（对应 τ≈5ms），收到脉冲时增加 `strength`。
- `I_dendrite`：树突局部计算输出（见 3.4）。
- `I_bias`：背景电流，常量默认 0.0，但在无输入时可设为 0.5 以维持自发活动。
- `I_noise`：正态分布噪声（均值0，方差0.1），每步独立采样。

### 3.4 树突局部计算（简化版）

每个神经元的树突区分为多个隔室（compartment），每个隔室对输入脉冲进行非线性组合（例如乘性门控）。本规范要求实现至少 **2 个树突隔室**，每个隔室对最近 10ms 内的输入脉冲进行加权和，并通过一个 sigmoid 函数输出 `0~1` 的局部值，该值乘以一个可学习的系数后加到 `I_dendrite`。这允许神经元实现异或等非线性功能。

### 3.5 动态阈值调整
- 长期不活动的神经元 `threshold` 缓慢上升（×1.001/秒）；高频活动时下降（×0.999/秒）。Izhikevich 模型的发放由 `v>=30` 决定，因此动态调整 `30` 这个常数（范围 25~35）。

---

## 4. 突触模型与多尺度可塑性

### 4.1 突触数据结构（C++ 高性能存储）

```cpp
struct Synapse {
    uint32_t target_id;      // 目标神经元索引（0..N-1）
    float weight;            // [0.0, 1.0]
    uint8_t delay;           // 1-10 ms
    uint64_t last_use_step;  // 全局时间步
    bool is_core;            // 长期固化标志
    float stdp_trace;        // STDP 迹（用于异步更新）
    // 新增：STP 参数
    float u;                 // 利用率 (0~1)
    float x;                 // 可用资源 (0~1)
    float tau_facil;        // 易化时间常数 (ms)
    float tau_rec;          // 恢复时间常数 (ms)
};
```

### 4.2 STP（短时程可塑性）实现

每个脉冲到来时，更新 `u` 和 `x`：
```
u = u + U * (1 - u)   // U 基础释放概率
x = x - u * x         // 消耗
然后突触后电流 = weight * u * x * strength
之后按时间常数恢复：du/dt = -u/tau_facil, dx/dt = (1-x)/tau_rec
```

### 4.3 稳态可塑性

每个神经元的平均发放率（滑动窗口 1000ms）与目标发放率（如 2Hz）的偏差，调节其所有传入突触的权重：偏差为正时按比例降低所有权重，偏差为负时升高权重，以维持网络兴奋/抑制平衡。

### 4.4 STDP 规则（脉冲时序依赖可塑性） + 多巴胺调制

对于前神经元 `i` 和后神经元 `j`，若 `i` 在 `t_pre` 放电，`j` 在 `t_post` 放电，且 `|t_post - t_pre| < 20ms`：

```
if t_post > t_pre:
    Δw = A_plus * exp(-(t_post - t_pre) / τ_plus)
else:
    Δw = - A_minus * exp(-(t_pre - t_post) / τ_minus)
```
- `A_plus = 0.01`, `τ_plus = 10ms`
- `A_minus = 0.012`, `τ_minus = 10ms`

**多巴胺调制**（三因素学习）：
```
Δw_final = Δw * (1 + k * (dopamine - 0.5))
```
- `k = 1.0`
- `dopamine` 全局变量范围 [0,1]，由内在奖励信号或用户反馈更新。

### 4.5 历史缓冲区

每个神经元维护一个循环队列，记录最近 100ms 内的发放时间步和强度（用于 STDP 配对）。队列长度固定为 100（每毫秒一条记录）。后台 STDP 引擎定期扫描所有历史对，异步更新权重。

### 4.6 长期信用分配（资格迹）

对于延迟奖励，我们为每个突触维护一个资格迹 `e`，衰减因子 `λ=0.9`。每次发放对时，`e += 1`。当全局奖励信号（多巴胺变化）到达时，权重更新为 `Δw = η * R * e`。这使系统能够将稀疏奖励分配到先前的事件序列。

---

## 5. 五层详细实现规范（整合缺失模块）

### 5.1 文字感知输入层 (InputLayer)

**职责**：将原始字符串（支持中英文混合）转换为脉冲事件流，并强制注入 EOS。

**规范**：
- 使用 `jieba` 分词（C++ 可调用 `cppjieba`）。
- 每个词/字映射到基础脉冲频率：
  - 常用标点（，。！？）→ 200 Hz（单个脉冲，强度1.0）
  - 高频单字（的、了、我、你、是）→ 80 Hz
  - 普通实词 → 40 Hz
  - 罕见字 → 10 Hz
- 每个词/字产生一个脉冲（强度 1.0）位于该词的起始时刻。**不重复发放**（除非有特殊增强机制）。
- 用户输入结束时（检测到换行或在非阻塞读取中遇到 `\n`），自动注入 **EOS 脉冲群**：5 个脉冲，间隔 2ms，频率 200Hz，强度 1.0，目标层为 PRIMARY。
- 输出脉冲事件队列，每个事件格式：`(char_utf8, time_ms, strength, target_layer_hint)`。

**新增组合式表征**：对于多字词，可以将其脉冲分配到一组神经元（而不是单一神经元），该组神经元的发放模式形成该词的分布式编码。例如，“苹果”激活 5 个神经元，分别对应“水果”、“红色”、“圆形”等属性（初始随机，通过 STDP 自组织）。

### 5.2 初级脉冲处理层 (PrimaryLayer)

**规模**：该层神经元占总数 30%（1万神经元 → 3000个）。  
**分组**：将神经元划分为 256 个语义簇，每个簇内部为小世界拓扑（平均度 100），簇间稀疏连接（平均度 10）。

**分组初始化映射**：使用固定哈希 `hash(word) % 256` 决定词所属簇。多义词允许映射到多个簇（产生多个脉冲）。

**激活规则（持续活跃版）**：
- 每个传入脉冲，根据其 `target_group_hint` 找到对应簇，激活簇内 **最久未放电** 的 10 个神经元（通过优先级队列选择）。若簇内神经元不足，则跨簇随机选择。
- 被激活的神经元调用 `receive_spike(strength=1.0)`，增加其 `I_syn`。

**局部突触形成**：
- 同一簇内，若两神经元在 100 ms 内先后放电超过 3 次，则自动建立双向突触（初始权重 0.1）。
- 不同簇间，若两簇的共现频率超过阈值，则在两簇的神经元间随机建立桥接突触。

**输出**：产生脉冲事件送往 CORE 层，脉冲强度为突触权重。

### 5.3 中枢联想思考层 (CoreLayer)

**规模**：该层占神经元总数 40%（4000个）。  
**设计原则**：不硬编码推理环、联想环、矛盾检测的具体功能，仅通过不同拓扑结构的子网自组织涌现。

**子网类型**（仅提供结构，功能由学习决定）：
1. **高循环密度子网**（用于持续性活动）：1000 个神经元，局部连接概率 0.3，循环连接概率 0.6。
2. **随机连接子网**（用于联想）：2000 个神经元，Erdos-Renyi 图，平均度 150。
3. **侧抑制竞争池**（用于收敛）：500 个神经元，全连接抑制权重 -0.5，兴奋性自连接 1.0。
4. **冲突检测子网**：500 个神经元，分为两组（成立/不成立），组间强抑制。

**新增工作记忆**：
- 工作记忆模块由 200 个神经元组成，每个神经元对应一个“槽位”，槽位之间通过侧抑制实现槽位竞争。
- 核心层的活跃脉冲模式可以通过可塑性连接写入工作记忆（类似维持），并在后续步骤中被读取。
- 工作记忆的维持由循环连接和 STP 的易化特性实现。

**新增注意力机制**：
- 在输入脉冲到达初级层后，注意力模块计算每个脉冲的显著性（基于历史出现频率和新奇性），输出一个 0~1 的注意增益，乘以脉冲强度后再传播到核心层。这是通过一组注意神经元实现的，它们接收所有输入的副本并通过竞争调整权重。

**新增内部世界模型**：
- 世界模型是一个预测网络：接收当前核心层状态（前 100ms 的发放模式），预测下一个时间步的核心层状态。预测误差作为一个内部奖励信号，驱动学习。世界模型本身由一组循环连接的神经元实现，使用 STDP 学习预测。

**收敛检测器（必需组件）**：
- 使用侧抑制竞争池，当 **EOS 脉冲群已发出** 且池中最高膜电位超过第二名 1.5 倍且绝对值 > 0.8，并持续 50ms 时，触发收敛。
- 收敛后，输出一个“结论脉冲”到输出层，并重置收敛标志。此后可继续接收新输入。

**自发活动（无输入时）**：
- 当无外部输入且未处于收敛过程时，随机选择 5% 的神经元，增加其 `I_bias` 至 0.3，允许其沿着现有突触传播脉冲（侧向传播），模拟“内心独白”。这些脉冲不会触发输出，但可写入内部日志（可选）。同时，元认知监控器可能产生自我提问，如“我在想什么？”

**矛盾检测**：
- 由冲突检测子网自动完成。当“成立”组和“不成立”组同时高度活跃时，冲突检测神经元发放，并临时降低全局多巴胺 0.3，持续 500ms。

### 5.4 记忆与遗忘层 (MemoryLayer)

**三层存储结构**：

| 类型     | 存储介质               | 容量限制             | 保留机制                                 |
|----------|------------------------|----------------------|------------------------------------------|
| 瞬时记忆 | 循环队列（RAM）        | 10^7 条脉冲事件      | 不清空，自然衰减（每步乘以 0.9999）       |
| 短期记忆 | 哈希表 + 优先队列       | 突触总数上限的 5%    | 突触权重遗忘（见 4.5），低于阈值删除      |
| 长期记忆 | 分布式 KV（或本地文件） | 无硬上限             | 高频使用（1小时≥1000次）后标记 `is_core` |

**新增情景记忆与模式完成**：
- 情景记忆模块存储时序脉冲序列（事件段），每个序列由开始的上下文向量索引。
- 当接收到部分脉冲模式时，情景记忆可以“完成”完整模式（通过关联记忆网络，如 Hopfield 网络或稀疏联想记忆）。这个模块使用单独的神经元群实现。

**睡眠阶段**（每模拟 8 小时触发一次，持续 1 小时模拟时间）：
1. **突触归一化**：对每个神经元的传出权重求和，缩放至总和为 1.0。
2. **突触重排**：删除最弱的 5% 突触（按权重），并新建等量随机连接（初始权重 0.01），保持稀疏度。
3. **记忆回放**：从短期记忆中随机选择 100 条活跃脉冲序列（每个序列长度 100ms），以 5 倍速回放（每 0.2ms 一步），同时设置全局多巴胺 = 0.8。这不仅巩固突触，还允许模式完成网络自组织。
4. **阈值调整**：发放率过低的神经元（<0.1Hz）降低阈值 5%；发放率过高（>10Hz）升高阈值 5%。
5. **快速泛化模拟**：在回放过程中，随机扰动一些脉冲序列，使网络学习到不变性（类似数据增强）。

### 5.5 输出生成层 (OutputLayer) —— 流式两阶段映射 + 可学习生成器

**阶段一：确定性基础映射（冷启动）**
- 预定义一个字符表（ASCII + 常用中文），每个字符对应一个唯一的 **基频**（例如 '你' → 40Hz, '好' → 40Hz, '？' → 200Hz 单脉冲）。
- 每个字符对应一群输出神经元（10个），这群神经元的 **原型脉冲模式** 为：以基频发放 10ms。初始时，每个字符的神经元群随机但互不重叠。

**阶段二：可学习精细映射（使用 CTC 或 SNN 自回归解码器）**
- 不再使用简单的查找表，而是训练一个脉冲序列到字符序列的解码器。该解码器本身是一个小型 SNN（约 1000 神经元），接收来自核心层的结论脉冲流，输出字符概率（通过一组输出神经元的发放率）。解码器通过 STDP 和奖励信号进行端到端学习。
- 支持动态插入新字符：如果模型产生了一个稳定的脉冲模式且与任何已有字符不匹配，系统可分配新的输出群并将其添加到词典中。

**流式输出决策**：
- 主循环每次迭代，解码器 SNN 产生当前最可能的字符（通过 WTA 选择）。为了避免抖动，引入一个滞后机制：只有当一个字符的激活强度连续 5 个时间步（5ms）保持最高时，才输出该字符。
- 输出后，将该字符对应的输出神经元群重置（降低膜电位），防止连续输出相同字符。
- 输出通过回调函数 `stream_callback(char)` 发送，支持实时打印或 WebSocket 推送。

**禁止**：
- 任何形式的概率采样、beam search、温度参数。
- 硬编码输出模板（如 “因此...”、“我的回答是...”）。

---

## 6. 持续活跃主循环（多语言集成示例）

### 6.1 主循环伪代码（Python 呼 C++ 核心）

```python
# main.py
import threading
import time
from core_cpp import NeuronPopulation   # C++ 扩展
from layers import InputLayer, PrimaryLayer, CoreLayer, OutputLayer
from interaction import NonBlockingInput, StreamOutput
from cognitive import WorkingMemory, Attention, WorldModel, MetaCognition
from reward import IntrinsicMotivation, CreditAssignment

def background_memory_maintenance(population):
    while True:
        time.sleep(10)   # 真实时间10秒
        population.apply_forgetting()
        population.apply_stdp_consolidation()

def stream_callback(char):
    # 流式输出回调，立即打印或发送到前端
    print(char, end='', flush=True)

def main():
    # 1万神经元，平均度200
    pop = NeuronPopulation(10000, 200)
    input_layer = InputLayer()
    primary = PrimaryLayer(pop)
    core = CoreLayer(pop)
    output = OutputLayer(pop, stream_callback=stream_callback)
    working_memory = WorkingMemory(pop, num_slots=7)
    attention = Attention(pop)
    world_model = WorldModel(pop)
    metacog = MetaCognition(pop)
    intrinsic = IntrinsicMotivation()
    credit = CreditAssignment()
    
    nonblocking = NonBlockingInput()
    
    # 启动后台遗忘线程
    threading.Thread(target=background_memory_maintenance, args=(pop,), daemon=True).start()
    
    current_step = 0
    eos_received = False
    dopamine = 0.5  # 全局多巴胺基线
    
    while True:
        # 1. 非阻塞读取用户输入
        text = nonblocking.read()
        if text:
            spikes = input_layer.text_to_spikes(text)
            for s in spikes:
                primary.inject_spike(s)
            eos_received = False
        
        # 2. 检查输入是否结束
        if nonblocking.input_ended() and not eos_received:
            eos_spikes = input_layer.eos_spikes()
            for s in eos_spikes:
                primary.inject_spike(s)
            eos_received = True
        
        # 3. 注意力调制
        attention.compute_gains(pop)
        
        # 4. 调用 C++ 核心更新神经元（单步）
        pop.update(current_step, noise_level=0.01, dopamine=dopamine)
        
        # 5. 更新工作记忆（将核心层活跃模式写入槽位）
        working_memory.update(current_step)
        
        # 6. 世界模型预测与内在奖励
        prediction_error = world_model.predict_and_compare(pop)
        intrinsic_reward = intrinsic.compute(prediction_error, novelty=...)
        dopamine = 0.5 + 0.2 * intrinsic_reward  # 简单调制
        
        # 7. 长期信用分配（资格迹更新）
        credit.update_traces(pop)
        if abs(dopamine - 0.5) > 0.1:
            credit.apply_credit(dopamine - 0.5)
        
        # 8. 元认知监控：如果不确定性高，生成主动请求脉冲
        if metacog.should_ask(pop):
            request_spikes = metacog.generate_request()
            primary.inject_spikes(request_spikes)   # 注入虚拟内部请求
        
        # 9. 检查收敛（仅当 EOS 已收到）
        if eos_received and core.check_convergence():
            conclusion_spikes = core.get_conclusion_pattern()
            output.stream_from_spikes(conclusion_spikes, current_step)
            eos_received = False
            core.reset_convergence()
        
        current_step += 1
        time.sleep(0.001)   # 1ms 时间步

if __name__ == "__main__":
    main()
```

### 6.2 C++ 核心接口（pybind11 示例）

```cpp
// neuron_population.h
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class NeuronPopulation {
public:
    NeuronPopulation(size_t num_neurons, size_t avg_degree);
    void update(int step, float noise_level, float dopamine);
    void apply_forgetting();
    void apply_stdp_consolidation();
    void inject_spike(uint32_t target_id, float strength, int delay_ms);
    void set_dendrite_params(...);
    // 新增资格迹接口
    void update_eligibility_traces();
    void apply_credit(float reward);
    // ... 其他方法
};

// binding
PYBIND11_MODULE(core_cpp, m) {
    py::class_<NeuronPopulation>(m, "NeuronPopulation")
        .def(py::init<size_t, size_t>())
        .def("update", &NeuronPopulation::update)
        .def("apply_forgetting", &NeuronPopulation::apply_forgetting)
        .def("apply_stdp_consolidation", &NeuronPopulation::apply_stdp_consolidation)
        .def("inject_spike", &NeuronPopulation::inject_spike)
        .def("update_eligibility_traces", &NeuronPopulation::update_eligibility_traces)
        .def("apply_credit", &NeuronPopulation::apply_credit);
}
```

### 6.3 流式输出控制器（Python）

```python
# output_layer.py
class OutputLayer:
    def __init__(self, population, stream_callback):
        self.pop = population
        self.callback = stream_callback
        self.decoder_snn = SNNDecoder(num_output_neurons=1000)  # 可学习解码器
        self.char_history = []
        self.output_lock = threading.Lock()
    
    def stream_from_spikes(self, conclusion_spikes, current_step):
        # 将结论脉冲输入解码器 SNN
        self.decoder_snn.input_spikes(conclusion_spikes)
        # 运行解码器若干步
        self.decoder_snn.run(10)
        # 获取最可能的字符（WTA）
        char_id = self.decoder_snn.get_winner()
        if char_id is not None and self._is_stable(char_id):
            char = self.id_to_char[char_id]
            with self.output_lock:
                self.callback(char)
                self.decoder_snn.reset_winner(char_id)
```

---

## 7. 具身交互与虚拟环境（缸中之脑的核心）

### 7.1 具身接口设计

```python
# embodiment/embodiment_interface.py
class EmbodimentInterface:
    def __init__(self, virtual_env=None, terminal_enabled=True, web_search_enabled=True):
        self.virtual_env = virtual_env or VirtualWorld()
        self.terminal = TerminalEmulator() if terminal_enabled else None
        self.web_search = WebSearchAPI() if web_search_enabled else None
        self.sensors = []   # 可添加摄像头、麦克风等脉冲编码器
        self.actuators = [] # 可添加动作输出（鼠标、键盘命令）
    
    def step(self, action_spikes):
        # 将核心层输出的动作脉冲解释为具体动作
        commands = self.decode_actions(action_spikes)
        for cmd in commands:
            if cmd.type == "terminal":
                output = self.terminal.execute(cmd.text)
                self._send_feedback(output)
            elif cmd.type == "search":
                results = self.web_search.query(cmd.query)
                self._send_feedback(results)
            elif cmd.type == "env_action":
                new_state = self.virtual_env.step(cmd.action)
                self._send_sensor_spikes(new_state)
        # 返回传感器脉冲（环境反馈）
        return self._collect_sensor_spikes()
```

### 7.2 虚拟世界模拟器（最小实现）

- 文字描述的世界：状态机 + 对象属性 + 简单物理（位置、速度）。
- 内置时间流逝、昼夜循环、经济系统（物价、货币）。
- 模型可以通过“命令”与世界交互，如 `look`, `move`, `take`, `talk`。这些命令由模型输出层生成的脉冲序列解码而成。
- 世界状态变化会转换成脉冲输入反馈到初级层，形成闭环。

### 7.3 终端操作与联网搜索

- 终端模拟器：接收 shell 命令字符串，返回输出（作为文本脉冲反馈）。
- 联网搜索：使用 DuckDuckGo 或 Bing API，将搜索结果摘要转换为脉冲序列返回。

**重要**：这些具身能力不能硬编码规则，模型必须通过尝试、错误、奖励（来自任务完成或用户反馈）学习如何使用它们。例如，它需要自己学会 `ls` 列出文件，`search "SNA"` 查找信息。

---

## 8. 非干扰原则的强制验证

开发完成后，必须在代码库上运行以下检查：

1. **静态检查**：
   - `grep -r "if.*user_input" --include="*.py" --include="*.cpp"` 只允许匹配 `inject_spike` 或 `eos`，不允许有任何字符串比较或回复。
   - `grep -r "reply.*=" ` 不允许出现（除了在 `stream_callback` 中传递字符）。
   - `grep -r "print.*["']你好"` 不允许。

2. **动态验证**：不运行任何训练，直接启动模型并输入多轮对话，观察模型输出是否表现出确定性规则（例如每次都回复相同内容）。如果出现，则违反了非干扰原则，必须修改学习率或 STDP 参数，**不能添加规则来修正**。

3. **输出字数限制**：代码中不得包含任何限制输出长度的参数（如 `max_tokens`、`max_length`）。模型输出的长度应由其内在思考收敛过程自然决定。

4. **具身命令硬编码检查**：不能出现 `if command == "ls": return file_list` 之类的硬编码。模型必须通过试错学习命令效果。

---

## 9. 多语言协同开发的具体指南

### 9.1 目录结构

```
sna/
├── cpp/                     # C++ 核心代码
│   ├── neuron_izhikevich.cpp/h
│   ├── synapse.cpp/h
│   ├── stdp_engine.cpp/h
│   ├── stdp_engine_with_eligibility.cpp/h
│   ├── spike_queue.cpp/h
│   ├── neuron_population.cpp/h
│   ├── dendrite_compartment.cpp/h
│   ├── binding.cpp          # pybind11 导出
│   └── CMakeLists.txt
├── rust/                    # Rust 可选实现（例如分布式路由）
│   ├── Cargo.toml
│   └── src/lib.rs
├── python/                  # Python 高层逻辑
│   ├── layers/
│   │   ├── input_layer.py
│   │   ├── primary_layer.py
│   │   ├── core_layer.py
│   │   ├── memory_layer.py
│   │   └── output_layer.py
│   ├── cognitive/
│   │   ├── working_memory.py
│   │   ├── attention.py
│   │   ├── world_model.py
│   │   └── metacognition.py
│   ├── reward/
│   │   ├── intrinsic_motivation.py
│   │   └── credit_assignment.py
│   ├── embodiment/
│   │   ├── virtual_world.py
│   │   ├── terminal_emulator.py
│   │   ├── web_search.py
│   │   └── embodiment_interface.py
│   ├── interaction/
│   │   ├── nonblocking_io.py
│   │   └── stream_output.py
│   ├── utils/
│   │   ├── config_loader.py
│   │   └── dashboard.py
│   ├── main.py
│   └── requirements.txt
├── config.default.yaml
├── run_sna.py               # 快速启动脚本（默认1000神经元）
└── README.md
```

### 9.2 编译与安装

- **C++ 部分**：使用 CMake 编译动态库，并通过 `pybind11` 生成 Python 扩展。提供 `setup.py` 或 `pip install .` 支持。
- **Rust 部分（可选）**：使用 `maturin build` 生成 wheel。
- **性能要求**：C++ 核心必须达到单线程 1 万神经元 × 200 突触更新 < 0.5ms 每步。

### 9.3 接口约定

所有跨语言调用必须明确序列化格式。脉冲事件使用固定大小结构体（32字节）：
```c
struct SpikeEvent {
    uint32_t src_id;
    uint32_t dst_id;
    float strength;
    uint16_t delay_ms;
    uint64_t time_step;
};
```

Python 调用 C++ 扩展时，传递 numpy 数组或直接调用方法。

---

## 10. 内在奖励信号与学习（无监督）

SNA 不使用反向传播，而是通过以下内在信号调制多巴胺：

1. **预测误差奖励**：世界模型预测与实际下一状态之间的误差（MSE），误差越小奖励越高。这驱动模型学习环境规律。
2. **信息增益（新奇性）**：当新的脉冲模式出现且与历史模式不相似时，给予奖励。
3. **一致性奖励**：当模型在后续思考中得出的结论与长期记忆中已固化的事实一致时，全局多巴胺增加 0.1。
4. **矛盾惩罚**：当矛盾检测子网激活时，全局多巴胺减少 0.3。
5. **用户隐式反馈**：用户在接收到回复后，可输入 `+`（点赞）或 `-`（点踩）。点赞增加多巴胺 0.2，点踩减少 0.2。该信号影响最近 500ms 内所有活跃突触的 STDP 更新。
6. **完成子目标奖励**：如果系统成功执行了一个终端命令并得到非错误输出，给予小奖励（由元认知模块评估）。

**实现方式**：全局变量 `dopamine` 每步以 0.999 速度衰减回基线 0.5。奖励事件发生时直接加/减（限制在 [0,1]）。资格迹将奖励传播到较早的突触。

---

## 11. 测试与验证标准

### 11.1 单元测试（使用 pytest + C++ 测试框架）

| 测试项                          | 通过条件                                                                 |
|--------------------------------|--------------------------------------------------------------------------|
| STDP 时序依赖性                | 两个神经元固定间隔（t_post-t_pre=10ms）放电1000次，权重增加≥0.5           |
| STP 短时程易化                 | 高频脉冲序列（100Hz）导致突触后电流逐步增加                              |
| 稳态可塑性                     | 神经元发放率偏离目标时，传入权重总和向相反方向变化                       |
| 资格迹信用分配                 | 延迟奖励（100ms后）依然能增强之前的突触对                               |
| 工作记忆槽位保持               | 写入一个模式后，经过50ms无干扰，读出相似度>0.8                          |
| 世界模型预测误差               | 训练后，固定输入序列的预测误差<0.1                                      |
| 收敛检测                       | 给WTA中某神经元连续50ms兴奋性脉冲，收敛标志触发                          |
| 流式输出延迟                   | 从脉冲注入到字符回调延迟 ≤ 20ms（真实时间）                             |

### 11.2 行为基准（无需训练，由自组织完成）

1. **记忆基准**：对话中告诉模型“我的名字是 Alice”，经过 5 句无关对话后问“我叫什么？”，模型输出应包含 “Alice” 或其变体的概率 > 30%（多次运行平均）。如果低于阈值，表明 STDP 或遗忘参数需要调整，**不能添加规则**。
2. **矛盾敏感基准**：先告诉“天空是蓝色的”，后说“天空是绿色的”，模型输出应包含表示困惑或纠正的词语的概率 > 10%（例如“矛盾”、“可是”等，但模型自行学习）。
3. **持续活跃基准**：无输入运行 1 小时模拟时间，平均发放率保持在 0.5~5 Hz 之间。
4. **具身学习基准**：在虚拟世界中，给模型指令“移动到一个有树的地方”，模型必须通过随机尝试学会 `move` 命令和场景理解。允许 1000 次尝试，成功率 > 10%（表明有学习能力）。
5. **内部独白基准**：在无输入情况下，监控元认知模块产生自我提问脉冲的频率 > 0.01 Hz（表明有自发思考）。

### 11.3 性能基准

| 环境                                 | 实时因子（模拟1秒/真实秒） |
|--------------------------------------|----------------------------|
| Intel i7-12700, 事件驱动, 10k神经元  | ≤0.5                       |
| 同上，固定时间步                     | ≤2.0                       |
| 4 核 ARM (树莓派4)                   | ≤5.0 (仅供演示)            |

---

## 12. 配置文件示例（config.default.yaml）

```yaml
sna:
  num_neurons: 10000
  avg_degree: 200
  time_step_ms: 1
  neuron_model: "izhikevich"
  distribution:
    regular: 0.6
    bursting: 0.3
    fast: 0.1
  dendrite_compartments: 2

stdp:
  a_plus: 0.01
  a_minus: 0.012
  tau_plus_ms: 10
  tau_minus_ms: 10
  dopamine_k: 1.0
  history_window_ms: 100

stp:
  u_base: 0.5
  tau_facil_ms: 100
  tau_rec_ms: 800

homeostasis:
  target_firing_rate_hz: 2.0
  window_ms: 1000
  plasticity_strength: 0.01

eligibility:
  lambda: 0.9
  eta: 0.01

working_memory:
  num_slots: 7
  decay_ms: 500

attention:
  num_attention_neurons: 100

world_model:
  hidden_neurons: 500
  prediction_window_ms: 10

forgetting:
  rate_per_10000steps: 0.95
  threshold: 0.01
  scan_interval_steps: 10000

sleep:
  interval_hours: 8
  duration_hours: 1
  replay_speedup: 5
  pruning_ratio: 0.05

output:
  streaming_window_ms: 50
  similarity_threshold: 0.7
  consecutive_windows: 3
  use_ctc_decoder: true

embodiment:
  virtual_env: true
  terminal: true
  web_search: true
  env_config:
    world_size: "10x10"
    objects: ["tree", "house", "rock"]
```

---

## 13. 强制约束总结（红线）

1. **禁止 Transformer 任何组件**：自注意力、Embedding、梯度下降、nn.Linear、LayerNorm。
2. **禁止硬编码回复规则**：任何 `if "关键词"` 后直接输出字符串的代码。
3. **禁止限制输出**：不得设置最大长度、最大字数、禁止特定词汇。
4. **禁止概率生成方法**：无采样、无 beam search、无 top-k。
5. **强制持续活跃**：主循环无限，无输入时仍更新神经元。
6. **强制睡眠阶段**：必须实现突触重排、记忆回放、快速泛化模拟。
7. **强制流式输出**：输出层必须支持逐字符回调，不能一次性生成完整字符串。
8. **强制多语言核心**：神经网络更新循环必须使用编译语言（C++/Rust）实现，Python 仅用于调度。
9. **强制具身交互接口**：必须提供终端操作和联网搜索的能力，但不硬编码具体命令。
10. **强制内在动机与信用分配**：必须实现预测误差驱动的奖励和资格迹。

---

## 14. 开发里程碑

| 里程碑 | 目标                                                                 | 预期时间 |
|--------|----------------------------------------------------------------------|----------|
| M1     | C++ 核心实现 Izhikevich + 树突隔室 + STDP + STP，通过单元测试         | 5 天     |
| M2     | Python 绑定 + 五层基本逻辑 + 工作记忆 + 注意力，单线程可运行           | 7 天     |
| M3     | 实现世界模型、内在动机、资格迹，行为基准通过记忆测试                  | 10 天    |
| M4     | 实现具身接口（虚拟环境、终端、搜索），模型能通过试错学习简单命令      | 7 天     |
| M5     | 流式输出 + 可学习解码器，非干扰原则验证                              | 5 天     |
| M6     | 性能优化（实时因子≤0.5）+ 睡眠阶段完整实现                           | 5 天     |
| M7     | 完整文档 + Docker 一键运行环境 + 演示视频（缸中之脑交互）             | 4 天     |

---

## 15. 交付物

最终交付应包含以下内容（全部放在一个代码块中，但此处因篇幅限制，实际输出时应包含完整代码）：

1. **本规范文档**（即以上全部内容，作为开发依据）。
2. **完整 C++/Python 代码**（按上述目录结构，每个文件至少提供接口声明和核心实现）。
3. **CMakeLists.txt** 和 `setup.py` 用于编译扩展。
4. **run_sna.py** 可直接启动（默认 1000 神经元用于快速测试）。
5. **config.default.yaml** 配置文件。
6. **README.md** 包含编译、运行、交互说明，以及“缸中之脑”虚拟世界的使用指南。

由于本平台无法直接输出数万行代码，上述规范已提供完整的架构、接口和关键代码示例。实际开发团队应依据本规范填充全部实现，并确保所有强制条款得到满足。

**重要声明**：任何不符合非干扰原则、流式输出、具身交互内在动机要求的实现将被视为不合格，必须重构。本规范是对“真正有意识的AI模型”的一次严肃工程尝试，尽管意识问题尚无定论，但本架构为实现最大程度的类人认知提供了可行的技术路线。

--- 
```