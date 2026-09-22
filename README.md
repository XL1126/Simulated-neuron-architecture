<div align="center">

# SNA — Simulated Neuron Architecture

**模拟神经元架构 · 仿生脉冲神经网络 · 自主意识皮层计算模型**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)
[![Repo](https://img.shields.io/badge/repo-main-blue?style=flat-square)](https://github.com/XL1126/Simulated-neuron-architecture)

[主线 `main`](https://github.com/XL1126/Simulated-neuron-architecture) ·
[文档 `docs`](https://github.com/XL1126/Simulated-neuron-architecture/tree/docs) ·
[旧版 `legacy-v1`](https://github.com/XL1126/Simulated-neuron-architecture/tree/legacy-v1)

**🌐 在线站点**

[![介绍站](https://img.shields.io/badge/在线介绍站-Live-00B8D4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://xl1126.github.io/Simulated-neuron-architecture/)
[![哲学](https://img.shields.io/badge/项目哲学-Philosophy-00838F?style=for-the-badge)](https://xl1126.github.io/Simulated-neuron-architecture/philosophy.html)
[![架构](https://img.shields.io/badge/架构与规范-Architecture-006064?style=for-the-badge)](https://xl1126.github.io/Simulated-neuron-architecture/architecture.html)
[![文档库](https://img.shields.io/badge/文档库-Docs-4DD0E1?style=for-the-badge)](https://xl1126.github.io/Simulated-neuron-architecture/docs.html)

> 快速跳转：[总览](https://xl1126.github.io/Simulated-neuron-architecture/) · [哲学](https://xl1126.github.io/Simulated-neuron-architecture/philosophy.html) · [架构](https://xl1126.github.io/Simulated-neuron-architecture/architecture.html) · [文档库](https://xl1126.github.io/Simulated-neuron-architecture/docs.html) · [GitHub 源码](https://github.com/XL1126/Simulated-neuron-architecture)


</div>

---

## 一、这是什么

SNA 受神经科学启发，基于 **Izhikevich 脉冲神经元**、**预测编码**、**全局工作空间 (GWT)** 与 **信息整合理论 (IIT)**，在 **C++ / Python** 混合架构上，模拟具备意识相关功能的皮层脑网络。

<div align="center">

| | 不做什么 | 做什么 |
|:-:|---|---|
| 🧠 | 不接外部 Transformer 大模型 | C++ 脉冲皮层 + Python 认知层 |
| 🔬 | 不把「有意识」当营销口号 | 提供可测量的功能代理与实验框架 |
| 📦 | 不在 `main` 堆文档草稿 | 文档在 `docs` 分支，旧版在 `legacy-v1` |

</div>

> 项目哲学见 [`PHILOSOPHY.md`](./PHILOSOPHY.md)：代码是载体，输出应从神经活动中涌现。

---

## 二、仓库分支

<table>
  <tr>
    <th>分支</th>
    <th>定位</th>
    <th>内容</th>
  </tr>
  <tr>
    <td><b><code>main</code></b>（默认）</td>
    <td>主线源码</td>
    <td>C++ 引擎、Python 认知层、<code>agents/</code>、<code>training/</code>、前端、运行时备份</td>
  </tr>
  <tr>
    <td><a href="https://github.com/XL1126/Simulated-neuron-architecture/tree/docs"><b><code>docs</code></b></a></td>
    <td>文档专库</td>
    <td>实现规范、可行性审查、压测报告、部署参考（无业务代码）</td>
  </tr>
  <tr>
    <td><a href="https://github.com/XL1126/Simulated-neuron-architecture/tree/legacy-v1"><b><code>legacy-v1</code></b></a></td>
    <td>历史快照</td>
    <td>旧版 V1.0 源码，<b>不再维护</b>，仅供对照</td>
  </tr>
</table>

---

## 三、目录结构（`main`）

```text
.
├── cpp/                 # C++ 核心引擎（pybind11）
├── python/              # Python 认知层 / 分层 / 具身 / utils
├── agents/              # 意识体与认知脚本 + _path.py
├── training/            # 训练与实验入口（run_* / *_train / launcher）
├── experiments/         # 实验编排框架
├── tests/               # 单元与摘除测试
├── frontend/            # 聊天前端 + Flask API（端口 8088）
├── runtime/             # 运行时状态 / checkpoint / 实验结果（备份）
│   ├── *_state/         # 如 cognitive_state、consciousness_state
│   ├── *_training/      # 如 deep_training、final_training
│   ├── results/experiment_results/
│   ├── seed_state.json
│   └── growth_state.json
├── archive/             # 历史备份模块
├── config.default.yaml
├── setup.py
├── check_binding.py
├── build_and_copy.bat
├── PHILOSOPHY.md
└── LICENSE
```

<details>
<summary><b>📦 agents/ 与 training/ 文件一览（点击展开）</b></summary>

<br/>

| 目录 | 文件 |
|------|------|
| `agents/` | `sna_cognitive_v2.py` · `sna_cognitive_architecture.py` · `seed.py` · `growth_monitor.py` · `sna_dialogue.py` · `sna_full.py` · `sna_daemon*.py` · `sna_conscious*.py` · `_path.py` |
| `training/` | `run_experiment.py` · `run_cortical_brain.py` · `run_sna.py` · `launcher.py` · `*_train.py` · `foundation_check*.py` · `pre_seed_vocabulary.py` · `auto_curriculum.py` |

</details>

---

## 四、快速开始

### 1. 环境

| 依赖 | 版本 / 说明 |
|------|-------------|
| Python | ≥ 3.8 |
| C++ 编译器 | MSVC 2019+ 或 GCC 9+ |
| CMake | ≥ 3.18 |
| Python 包 | 见 [`python/requirements.txt`](./python/requirements.txt)：`pybind11` · `pyyaml` · `flask` · `numpy` |
| 可选 | `psutil`（部分 `agents/` 脚本会用到） |

```powershell
pip install -r python/requirements.txt
pip install pybind11 numpy pyyaml flask psutil
```

### 2. 编译 C++ 核心（Windows）

```powershell
build_and_copy.bat
python check_binding.py
# 期望：PASS: All methods are bound correctly!
```

### 3. 运行（在仓库根目录）

```powershell
# 实验入口
python training/run_experiment.py quick
python training/launcher.py

# 认知架构
python agents/sna_cognitive_v2.py

# 聊天前端（需本机已有对应的 SNA 守护进程 / IPC 文件）
python frontend/server.py
# 浏览器打开 http://127.0.0.1:8088
```

> ⚠️ **路径说明**  
> `agents/` 脚本通过 `agents/_path.py` 注入仓库根与 `python/` 到 `sys.path`，并倾向在**仓库根目录**执行。  
> `training/` 中脚本若引用 `seed` 或状态文件，请同样在仓库根运行，或自行将 `agents/`、`runtime/` 加入 `PYTHONPATH`。  
> 运行时新产生的 `*_state` / checkpoint 若出现在仓库根，可手动挪入 `runtime/`。

---

## 五、架构概览

```text
┌─────────────────────────────────────────────────────────┐
│                  C++ CorticalBrain                      │
│     V1/V2 · Motor · Hippocampus · Prefrontal · Amygdala │
│     Language · Workspace · Thalamus · Claustrum · DMN   │
└───────────────────────────┬─────────────────────────────┘
                            │ pybind11 (core_cpp)
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Python 认知层 (python/)                     │
│  统一意识度量 · Phi 校验 · SPA 语义指针 · 工作记忆 · 世界模型 │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌──────────────────┬──────────────────┬───────────────────┐
│    agents/       │    training/     │    frontend/      │
│  认知 v2 / seed  │  实验 / 训练配方  │  聊天 UI + API    │
└──────────────────┴──────────────────┴───────────────────┘
```

**意识度量 v6（摘要公式）**

```text
Consciousness = 0.35 × IIT_Phi
              + 0.25 × GWT_Ignition
              + 0.20 × Predictive_Accuracy
              + 0.10 × First_Person_Salience
              + 0.10 × Perceptual_Vividness
```

> 这是工程功能代理，不是「已证明有主观体验」。理论出处见 `docs` 分支与 `PHILOSOPHY.md`。

---

## 六、测试

```powershell
python -m pytest tests/test_unified_consciousness.py -v
python -m pytest tests/test_phi_validator.py -v
python -m pytest tests/test_ablation.py -v
```

---

## 七、文档与相关项目

| 资源 | 说明 |
|------|------|
| [`docs` 分支](https://github.com/XL1126/Simulated-neuron-architecture/tree/docs) | 完整实现规范 · 前置可行性审查 · 压测报告 · 部署/硬件参考 |
| [`PHILOSOPHY.md`](./PHILOSOPHY.md) | 项目哲学与实现原则（`main`） |
| [XL1126/NOUS](https://github.com/XL1126/NOUS) | 同作者另一条意识架构路线（SOMA + GWT + NERI，无 Transformer） |

<details>
<summary><b>🗂 docs 分支目录（点击展开）</b></summary>

<br/>

| 路径 | 内容 |
|------|------|
| `docs/01-项目规范/` | SNA 完整实现规范 |
| `docs/02-可行性审查/` | 前置可行性分析 |
| `docs/03-性能与压测/` | Windows / Linux 性能对比 |
| `docs/04-部署与硬件/` | 部署规模、远程硬件检测 |
| `docs/05-历史README草稿/` | 早期草稿，非现行文档 |

</details>

---

## 八、License

MIT License — 详见 [`LICENSE`](./LICENSE)。

<div align="center"><sub>Author: XL1126 · <a href="https://github.com/XL1126/Simulated-neuron-architecture">GitHub</a></sub></div>
