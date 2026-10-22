---
type: entity
tags:
  - paper
  - world-models
  - nvidia
  - cosmos
  - interactive-simulation
  - sim-to-real
  - autonomous-driving
  - openmdw
status: complete
updated: 2026-10-22
project_id: cosmos-interactive-world-simulation
project: https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/
related:
  - ./cosmos-3.md
  - ./nvidia-cosmos.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/cosmos_interactive_world_simulation_tech_report_2026.md
summary: "Cosmos Interactive World Simulation（2026-10，NVIDIA）：把 Cosmos 3 改造成可流式交互的世界仿真，开源 Bimanual、Transfer 与 Auto 三类 Nano 模型卡，覆盖机器人交互、sim-to-real 视频迁移与多视角驾驶仿真。"
---

# Cosmos Interactive World Simulation（交互式世界仿真）

## 一句话定义

**这是 NVIDIA Cosmos 3 的后训练分支：把视频世界模型改成按动作与场景控制流式生成观测，让具身智能体或仿真器驱动的虚拟环境可以实时交互。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AV | Autonomous Vehicle | 自动驾驶，报告的三类场景之一 |
| sim-to-real | Simulation-to-Real | 仿真控制到真实外观的视觉迁移（Transfer 模型） |
| MoT | Mixture-of-Transformers | 模型卡所列架构 |
| LiDAR | Light Detection and Ranging | Transfer-Auto 可同时生成的激光雷达距离图 |

## 为什么重要

- 世界模型若只能离线生成视频，难以直接作为具身智能体的交互环境；流式、可控的生成使其可用于策略评估与 RL 训练。
- 报告把机器人操作、视频迁移与驾驶仿真放进同一后训练框架，便于比较不同场景下的共享设计（据摘要）。

## 方法与机制

- **后训练配方：** 报告目录包括 Cosmos 3 骨干、多模态 token 排布、时序因果注意力，以及双向适配、因果训练（teacher forcing）与因果一致性蒸馏（据目录）。
- **输入输出：** 动作或场景控制流入，视觉观测流出；外部仿真器可保持物理状态的权威。
- **模型族：** Bimanual（动作条件动力学）、Transfer（空间控制 → RGB）、Auto（多视角驾驶，可含 LiDAR）。

## 实验与评测

- 报告摘要说明覆盖相机导航、机器人操作、可控视频迁移与多视角 AV 仿真。
- 模型卡数字（来自 Hugging Face 模型卡，经摘要核对）：
  - Bimanual 相机导航（PAI-Bench-G）：视频质量 73.9，对比 LingBot-World 2.0 的 65.4；绝对平移误差高于 LingBot-World 2.0（4.71 对 4.17）。
  - Bimanual AgiBot-World：视频质量 76.1，对比 DreamDojo 14B 的 74.9。
  - Transfer（PAI-Bench-C）：分割 mIoU 0.738 对比 Cosmos3-Nano 的 0.643；深度 SI-RMSE 0.619 对比 0.842（越低越好）。
  - Transfer-Auto（720p）：文生视频 Frame FID 45.5 对比 Cosmos-Transfer2.5-2B Multiview 的 61.9。

## 与其他工作对比

- 与 DreamDojo、LingBot-World 2.0 的对比见上方模型卡数字；模型卡称基线的提示词设置可能不同。
- 与 Cosmos 3 基座：Transfer 在控制一致性指标上提升，但 DOVER 与 prompt following 略低（模型卡）。

## 结论

本报告的核心判断是：同一个 Cosmos 3 后训练配方可以支撑交互式、可控的世界仿真，而不是只做离线视频生成。

- 流式与因果训练是交互性的关键，报告以此区别于离线视频模型。
- 控制保真（分割、深度、边缘）是 sim-to-real 视频迁移的主要改进点，画质指标并未全面提升。
- 证据主要来自模型卡与摘要；完整报告的实验表与附录未逐页核对。

## 工程实践

- 代码：`nvidia/cosmos` 与 `nvidia/cosmos-framework`（报告列出）。
- 模型：四个 Hugging Face 模型卡，许可 OpenMDW 1.1。
- 推理门槛：模型卡所列测试环境为 Linux，且需要 NVIDIA Blackwell 或 Hopper GPU；Transfer-Auto 单路 480p 视频约 1 分 22 秒（GB200，无 LiDAR）。

## 局限与风险

- 模型卡明确仅在 Linux 与 Blackwell / Hopper GPU 上测试。
- Bimanual 模型卡仅列 AgiBot 本体；其他机器人需要另行后训练。
- 生成画面与仿真器标注可能不一致，使用前需逐场景校验。
- 模型卡示例关闭了 guardrails；部署前需自行加安全措施（模型卡）。

## 关联页面

- [Cosmos 3](./cosmos-3.md) — 本报告的基座模型（arXiv 2606.02800）
- [NVIDIA Cosmos](./nvidia-cosmos.md) — Cosmos 家族总览
- [Generative World Models](../methods/generative-world-models.md) — 视频世界模型方法族

## 参考来源

- [Interactive World Simulation 技术报告归档](../../sources/papers/cosmos_interactive_world_simulation_tech_report_2026.md)
- [Cosmos 3 技术报告（arXiv:2606.02800）](./cosmos-3.md)

## 推荐继续阅读

- [项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/)
- [技术报告 PDF](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf)
- [Hugging Face 模型合集](https://huggingface.co/collections/nvidia/cosmos3-examples)
