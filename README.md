# SNA legacy-v1（旧版 V1.0 快照）

本分支保存 **SNA 旧版 V1.0** 源码，**仅供对照与考古，不再维护**。

- 主线代码：[main](https://github.com/XL1126/Simulated-neuron-architecture/tree/main)
- 设计文档：[docs](https://github.com/XL1126/Simulated-neuron-architecture/tree/docs)

## 内容

| 路径 | 说明 |
|------|------|
| cpp/ | V1 时期 C++ 皮层脑核心 |
| python/ | V1 Python 认知 / 分层 / 具身 |
| experiments/ | V1 实验框架 |
| 	ests/ | V1 测试 |
| experiment_results/ | V1 实验结果 JSON |
| 根目录脚本 | un_sna.py / un_cortical_brain.py / launcher.py 等 |

编译产物（.pyd / .so）不入库；如需运行请用 CMake / uild_and_copy.bat 重新构建。

## 说明

该快照对应 2026-05 前后的 V1 工程状态；之后的认知 v2、seed 系统等只在 main 分支。