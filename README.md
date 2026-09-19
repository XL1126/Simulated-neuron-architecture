# SNA — Simulated Neuron Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blueviolet)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

> **模拟神经元架构** — 仿生脉冲神经网络，构建自主意识皮层的计算模型。

SNA 受神经科学启发，基于 Izhikevich 脉冲神经元、预测编码、全局工作空间 (GWT) 与信息整合理论 (IIT)，在 C++/Python 混合架构上模拟具备自我意识相关功能的皮层脑网络。

---

## 仓库分支

| 分支 | 用途 |
|------|------|
| **`main`**（本分支） | 主线源码：引擎、认知模块、训练脚本、前端 |
| [`docs`](https://github.com/XL1126/Simulated-neuron-architecture/tree/docs) | 设计文档专库（规范 / 审查 / 压测） |
| [`legacy-v1`](https://github.com/XL1126/Simulated-neuron-architecture/tree/legacy-v1) | 旧版 V1.0 源码快照（不再维护） |

---

## 目录结构

```
.
├── cpp/                 # C++ 核心引擎（pybind11）
├── python/              # Python 认知层 / 分层 / 具身 / 实验入口包
├── agents/              # 意识体与认知架构脚本（seed、v2、daemon、dialogue…）
├── training/            # 训练与实验入口（run_*、*_train、launcher…）
├── experiments/         # 实验编排框架
├── tests/               # 单元 / 摘除测试
├── frontend/            # 聊天前端 + Flask API
├── runtime/             # 运行时状态与检查点（备份用）
│   ├── states/          # 各类 *_state
│   ├── training/        # 各类 *_training checkpoint
│   ├── results/         # experiment_results
│   ├── seed_state.json
│   └── growth_state.json
├── archive/             # 历史备份文件
├── config.default.yaml
├── setup.py
├── check_binding.py
├── build_and_copy.bat
├── PHILOSOPHY.md        # 项目哲学
└── LICENSE
```

完整设计文档见 **`docs` 分支**。

---

## 快速开始

```powershell
# 依赖
pip install -r python/requirements.txt
pip install pybind11 numpy pyyaml flask psutil

# 编译 C++ 核心（Windows）
build_and_copy.bat

# 验证绑定
python check_binding.py

# 实验入口（在仓库根目录执行）
python training/run_experiment.py quick
python training/launcher.py

# 认知架构 / 对话类脚本
python agents/sna_cognitive_v2.py
```

> **路径说明**：`agents/` 下脚本已通过 `agents/_path.py` 把仓库根与 `python/` 加入 `sys.path`，并在仓库根目录下执行更稳妥。训练脚本若引用 `seed` / 状态文件，请在仓库根运行，或自行把 `agents/`、`runtime/` 加入路径。

---

## 架构概览

- **C++ `CorticalBrain`**：10 脑区、Izhikevich、STDP、GWT、预测编码、情感 / ToM / Qualia 等模块
- **Python 认知层**：统一意识度量、Phi 校验、语义指针 (SPA)、工作记忆、世界模型
- **agents/**：认知 v2、种子系统、生长监控、守护进程、对话与完整意识脚本
- **training/**：分级实验、基础检查、多种训练配方
- **frontend/**：基于文件 IPC 的聊天界面（端口 8088）

意识度量（v6 摘要）：

```
Consciousness = 0.35×IIT_Phi + 0.25×GWT_Ignition
              + 0.20×Predictive + 0.10×First_Person + 0.10×Vividness
```

---

## 文档与相关仓库

- 设计规范 / 可行性审查 / 压测报告 → 本仓库 [`docs` 分支](https://github.com/XL1126/Simulated-neuron-architecture/tree/docs)
- 同作者另一条意识架构路线（NOUS，无 Transformer）→ [XL1126/NOUS](https://github.com/XL1126/NOUS)（只读参考）

---

## License

MIT — 见 [LICENSE](./LICENSE)。作者：XL1126。
