# DriftWorld Project Page

> 来源归档

- **标题：** DriftWorld — Fast World Modeling through Drifting
- **类型：** site / project page
- **URL：** <https://susie-lu.github.io/driftworld/>
- **论文：** <https://arxiv.org/abs/2607.15065>
- **代码：** <https://github.com/Susie-Lu/driftworld>
- **权重：** <https://huggingface.co/Susie-Lu/driftworld>
- **入库日期：** 2026-07-22
- **一句话说明：** DriftWorld 官方项目页：展示 **1-step 动作条件 drifting 世界模型**、五环境视觉对比、GPC-RANK 推理时策略改进、离线策略评估相关性，以及 Language Table / Push-T **实时交互 demo**（单次前向，L4 GPU）。

## 开源状态（项目页核查，2026-07-22）

| 项 | 状态 |
|----|------|
| Paper | 已发布 — arXiv:2607.15065 |
| Code | 已挂链 — [Susie-Lu/driftworld](https://github.com/Susie-Lu/driftworld) |
| Checkpoints | 已发布 — HF `Susie-Lu/driftworld` |
| 复现范围 | **部分开源**：README 目前完整给出 **Push-T** 训练 / 可视化 / 指标 / GPC-RANK / 策略评估入口；Bridge-V2、RT-1、Language Table、Robomimic **Will be added soon** |
| License | GitHub 仓库元数据 **未声明** license（截至入库日） |

## 页面结构（策展）

- **Abstract / Overview** — 单次前向 30+ fps、平均约 **17×** 快于扩散 WM 基线
- **Demos** — Bridge-V2 / RT-1 动作条件生成示例
- **World Modeling 可视化** — 与 IRASim / WorldGym / Ctrl-World / GPC / LVDM 对照
- **Inference-Time Policy Improvement** — GPC-RANK 多提案想象选优
- **Policy Evaluation** — Lift / Can / Push-T 与真机/仿真 ground truth 相关性（最高约 **0.99**）
- **Interactive Demo** — Language Table / Push-T 实时键盘控制

## 对 wiki 的映射

- 论文归档：[`sources/papers/driftworld_arxiv_2607_15065.md`](../papers/driftworld_arxiv_2607_15065.md)
- 代码归档：[`sources/repos/driftworld.md`](../repos/driftworld.md)
- 沉淀 **[`wiki/entities/paper-driftworld.md`](../../wiki/entities/paper-driftworld.md)**
