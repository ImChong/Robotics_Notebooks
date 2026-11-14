# vla-evaluation-harness 官方源码导航（2026-11-14）

- **资料类型：** repo / 开源评测基础设施
- **英文名：** vla-evaluation-harness
- **中文：** 面向视觉语言动作模型的统一评测框架
- **代码：** https://github.com/allenai/vla-evaluation-harness
- **维护组织：** Allen Institute for AI（Ai2；GitHub `allenai`）
- **核对范围：** 官方 README 的介绍、News、Quick Start、Python API、Container/Docker、Benchmarks/Models 矩阵。
- **通读：** 未完成（已核对 README 关键章节和命令，未逐一审计后端模块）。
- **来源纠错：** 二手文章前文标「MAUM.AI」、后文写「Ai2」。以当前官方 `allenai` 仓库与 README 为准，不能把前文单处名称当维护归属证据。

## 可复现路径与边界

评测端**把模型 server 与 benchmark 配置分开**，支持容器隔离、分片运行、SQLite 录制与导出。官方 README 的命令：

- 安装：`pip install vla-eval`（Python 3.11+）。
- 模型服务：`vla-eval serve --config configs/model_servers/db_cogact/libero.yaml`。
- 基准评测：`vla-eval run --config configs/benchmarks/libero/smoke_test.yaml`。
- 训练内联评测：`vla_eval.evaluate(...)`（须按官方 Python API 创建 ModelServer）。

**开源状态：** CLI、配置、容器配方与示例已公开；仓库支持矩阵中的 ✓（复现）/ ◇（已集成待复现）必须区分，不能把已集成当成分数已复现。Docker 镜像的任务资源、模型权重可能另有许可证与下载要求。未运行评测。

## 对 wiki 的映射

- [2026 VLA 开源生态对比](../../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md)
- [2025 VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)

## 参考来源

- https://github.com/allenai/vla-evaluation-harness
- https://allenai.github.io/vla-evaluation-harness/
