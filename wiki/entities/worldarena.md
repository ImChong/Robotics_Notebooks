---
type: entity
tags: [benchmark, embodied-ai, video-generation, world-models, manipulation, evaluation, open-source, tsinghua]
status: complete
updated: 2026-10-26
summary: "WorldArena 1.0/2.0 评估具身世界模型的视频感知与功能效用；2.0 扩展到视触觉预测、在线 RL 环境和模拟/真机跨平台评估，榜单收录 Xiaomi-Robotics-U0。"
project_id: worldarena
arxiv: '2602.08971'
papers: ["2602.08971", "2605.17912"]
code_v2: https://github.com/WorldArena2/WorldArena-2.0
project: https://world-arena.ai/
code: https://github.com/tsinghua-fib-lab/WorldArena
related:
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ./ewmbench.md
  - ./worldolympiad.md
  - ./awesome-world-models.md
  - ./humanoid-motion-intelligence.md
sources:
  - ../../sources/repos/worldarena.md
  - ../../sources/sites/worldarena.md
  - ../../sources/papers/sun_awesome_wm_2602_08971_worldarena-a-unified-benchmark-for-evalu.md
  - ../../sources/papers/worldarena2_arxiv_2605_17912.md
  - ../../sources/sites/worldarena2.md
  - ../../sources/repos/worldarena2.md
---

# WorldArena（具身世界模型评测）

## 一句话定义

WorldArena 是具身世界模型统一评测套件：WorldArena 1.0 的 Track 1 测生成视频质量，Track 2 测模型作为数据引擎、策略评估器和动作规划器的功能价值，并以 EWMScore 汇总多维表现。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EWM | Embodied World Model | 面向具身任务的环境状态/未来预测模型 |
| EWMScore | Embodied World Model Score | WorldArena 汇总多项归一化指标的综合分 |
| IDM | Inverse Dynamics Model | 从状态变化估计动作的模型；用于动作规划评测 |
| OOD | Out-of-Distribution | 与训练分布不同的评测样本 |

## 为什么重要

- **画面逼真不等于策略有用。** WorldArena 将视频感知质量与下游功能评测放在同一套框架里，能暴露感知指标和实际任务效用之间的落差。
- **Track 1 可定位生成视频的具体失效类型。** 六个维度覆盖外观、运动、跨帧内容保持、物理交互、3D几何和指令/动作可控性。
- **Track 2 提供功能侧验证。** 它分别检查合成数据是否能提升策略、世界模型能否对策略排序、以及是否能支持动作规划；这些结果不应被 Track 1 的视频质量分替代。

## 核心结构

### WorldArena 1.0 两条赛道

| 赛道 | 测什么 | 公开评测方式 |
|------|--------|--------------|
| **Track 1：Video Quality** | 生成视频的视觉/运动质量、场景一致、物理合理、3D准确和可控性 | 提供首帧与文本指令或动作序列，生成对应视频，再运行官方指标管线；官方指南要求 24 fps，文本驱动固定 121 帧，分辨率建议至少 640×480 |
| **Track 2：Functional Performance** | 世界模型对策略学习和决策的效用 | Data Engine、Policy Evaluator；论文和项目页还描述 Action Planner 评测 |

Track 1 的 16 项基础指标归为六个维度：

| 维度 | 指标 |
|------|------|
| 视觉质量 | Image Quality、Aesthetic Quality、JEPA Similarity |
| 运动质量 | Dynamic Degree、Flow Score、Motion Smoothness |
| 内容一致性 | Subject Consistency、Background Consistency、Photometric Consistency |
| 物理遵循 | Interaction Quality、Trajectory Accuracy |
| 3D 准确性 | Depth Accuracy、Perspectivity |
| 可控性 | Instruction Following、Semantic Alignment、Action Following |

**EWMScore** 是 16 个归一化基础指标缩放到 0–100 后的算术平均。读分时仍需查看分项：单一总分会掩盖视频外观、动作遵循和空间稳定性的不同强弱。

### Track 1 复现流程

```mermaid
flowchart LR
  D[官方测试集与首帧] --> C[指令或动作条件生成]
  C --> V[对齐帧率与视频长度]
  V --> M[16项视频质量指标]
  M --> S[维度分数与EWMScore]
```

官方测试集基于 RoboTwin 2.0 Clean-50 配置。评测指南要求使用公布的测试集；文本驱动视频为 121 帧，动作驱动视频需与动作轨迹长度对齐。Challenge 1.0 已结束，官方评测资源已公开，可按仓库指南本地评测。


### WorldArena 2.0 扩展

WorldArena 2.0 将评测扩展到三个方向：

| 方向 | 1.0 到 2.0 的变化 | 读法 |
|------|-------------------|------|
| 模态 | 从视觉扩展到视触觉 | 测模型是否预测接触、力、滑移与材料交互，不止生成画面 |
| 功能 | 增加世界模型作为在线 RL 环境 | 检查模型能否稳定支持策略优化与交互 rollout |
| 平台 | 从仿真扩到仿真和真机 | 在 RoboTwin 2.0、LIBERO 与 AgileX ALOHA 等平台检查跨具身和 sim-to-real |

官方代码位于 [WorldArena 2.0 仓库](../../sources/repos/worldarena2.md)，榜单由 [Hugging Face Space](https://huggingface.co/spaces/WorldArena/WorldArena2.0) 展示。当前榜单包含 Xiaomi-Robotics-U0 条目；这是小米模型参加该基准评测的结果，WorldArena 2.0 本身由独立研究团队提出，不应写成小米自研基准。

## 工程实践

1. **先明确赛道。** 比较视频生成质量时按 Track 1 准备首帧、文本指令/动作和生成视频；验证对策略的价值时按 Track 2 准备对应模型适配器与 rollout 环境。
2. **按官方协议对齐输入输出。** Track 1 使用 24 fps；文本驱动固定 121 帧；动作驱动输出长度与给定轨迹一致。分辨率至少建议 640×480。
3. **保留分项结果。** 报告 EWMScore 时，同时列出六维指标，尤其是物理遵循、3D准确性和动作可控性。
4. **谨慎横向比较。** WorldArena、[EWMBench](./ewmbench.md) 和 [WorldOlympiad](./worldolympiad.md) 的数据、任务和裁判协议不同，不能直接用综合分高低给模型排通用名次。

## 局限与风险

- WorldArena 1.0 Track 1 基于 RoboTwin 仿真操作视频；成绩不能直接代表开放场景、真实相机或真机任务表现。
- EWMScore 汇总感知指标，不能取代策略闭环成功率、真机迁移或安全评测。
- **WorldArena 2.0 是独立扩展版**，增加视触觉、交互式 RL 环境和模拟/真机平台；不要把 2.0 榜单与 1.0 Track 1 结果当成同一协议。榜单出现 Xiaomi-Robotics-U0 只表示该模型参评，不代表基准归属小米。
- Challenge 1.0 提交已截止；本地复现以 GitHub 指南与当前公开测试资源为准。

## 关联页面

- [EWMBench](./ewmbench.md) — 聚焦机器人操作视频的场景、运动与语义质量。
- [WorldOlympiad](./worldolympiad.md) — 对长视频世界模型诊断物理、3D几何与长程交互。
- [生成式世界模型](../methods/generative-world-models.md) — 视频预测与生成模型的技术背景。
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 从预测保真度到功能效用的基准选型。
- [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) — 所在研究路线索引。

## 参考来源

- [WorldArena 官方论文与项目页](../../sources/sites/worldarena.md)
- [WorldArena 官方代码与 Track 1 评测指南](../../sources/repos/worldarena.md)
- [WorldArena 论文摘录](../../sources/papers/sun_awesome_wm_2602_08971_worldarena-a-unified-benchmark-for-evalu.md)
- Shang et al., [WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models](https://arxiv.org/abs/2602.08971)

## 推荐继续阅读

- [Track 1 指标实现与评测要求](https://github.com/tsinghua-fib-lab/WorldArena/tree/main/video_quality)
- [WorldArena 1.0 官方评测指南](https://github.com/tsinghua-fib-lab/WorldArena/blob/main/assets/README_submission.md)
- [WorldArena 官方项目页](https://world-arena.ai/)
