# AI 算子生成与性能评测资料

- **类型：** paper / GPU / NPU / kernel generation / benchmark
- **入库日期：** 2026-10-28
- **范围：** 以下四项为独立 benchmark，与 Humanize 仓库不构成已证实的项目关系。

## SOL-ExecBench（2026）

- **论文：** <https://arxiv.org/abs/2603.19173>
- **摘要要点：** 235 个 CUDA 优化问题源自 124 个模型，针对 NVIDIA Blackwell；SOLAR 推导硬件 Speed-of-Light 界限，SOL Score 衡量相对发布基线到硬件界限的差距被缩小多少。评测工具包含时钟锁定、L2 清理、进程隔离及静态检查。

## KernelBench（ICML 2025）

- **论文：** <https://arxiv.org/abs/2502.10517>
- **摘要要点：** 基于 250 个 PyTorch 工作负载，fast_p 统计功能正确且相对基线加速超过阈值 p 的生成 kernel 比例。执行和 profiling 反馈可用于迭代优化。

## MultiKernelBench（2025）

- **论文：** <https://arxiv.org/abs/2507.17773>
- **摘要要点：** 285 个任务、14 个类别，支持 NVIDIA GPU、华为 NPU 和 Google TPU；采用可扩展的后端抽象并研究类别内示例提示。

## CANN Bench（2026）

- **论文：** <https://arxiv.org/abs/2607.20518>
- **摘要要点：** 面向昇腾 NPU，当前论文所述版本覆盖 53 个算子、1060 个测试用例，评分分别考虑编译、正确性与性能；性能参考 PyTorch-on-Ascend 基线及逐用例硬件锚定性能界限。

## 对 wiki 的映射

- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)：对比正确性、软件基线和硬件界限的不同判分目标。
