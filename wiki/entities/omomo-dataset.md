---
type: entity
title: OMOMO（人–物交互动捕数据集）
tags: [dataset, mocap, human-object-interaction, manipulation, smpl-h, stanford, siggraph-asia-2023]
summary: "Stanford SIGGRAPH Asia 2023 人–物交互数据集：15 物体、约 10 h 全身操纵 MoCap（物体运动 + SMPL-H 人体），常被 OmniRetarget / ResMimic 等用作 G1 loco-manipulation 重定向源。"
updated: 2026-06-16
status: complete
related:
  - ../concepts/motion-retargeting.md
  - ./omniretarget-dataset.md
  - ./paper-hrl-stack-03-omniretarget.md
  - ./holosoma.md
  - ./amass.md
  - ./lafan1-dataset.md
  - ../comparisons/humanoid-reference-motion-datasets.md
sources:
  - ../../sources/repos/omomo_release.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
---

# OMOMO（Object Motion Guided Human Motion Synthesis）

**OMOMO** 是 Li et al.（Stanford，SIGGRAPH Asia 2023）发布的人–物交互（HOI）**动捕数据集与合成代码**：以 **物体运动序列为条件** 生成/记录人类全身操纵行为。公开集含 **15 类物体、约 10 小时** 交互动作，每条含 3D 物体几何、物体轨迹与人类全身姿态（SMPL-H / SMPL-X 生态）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OMOMO | Object MOtion guided human MOtion synthesis | 物体运动引导的人体全身合成与数据集 |
| HOI | Human-Object Interaction | 人–物交互，操纵与接触丰富任务 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| SMPL-H | SMPL with Hands | 带手部参数的 SMPL 人体模型 |
| G1 | Unitree G1 Humanoid | 宇树教育科研人形实验平台 |
| WBT | Whole-Body Tracking | 全身关节/根轨迹跟踪类 RL 任务 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |

## 为什么重要

- **loco-manipulation 上游**：与纯行走库不同，OMOMO 强调 **持物、推拉、搬运** 等物体耦合动作，是 [OmniRetarget](./paper-hrl-stack-03-omniretarget.md) HF 数据集 `robot-object/`（3.0 h）的主要来源。
- **工程样板数据**：[holosoma](./holosoma.md) 提供 `demo_omomo_wb_tracking.sh`，串联下载 → 重定向 → WBT 训练。
- **多工作共用**：ResMimic、VisualMimic 等在 GMT / 跟踪阶段与 [AMASS](./amass.md) 并列使用 OMOMO 扩充分布。

## 核心信息

| 字段 | 内容 |
|------|------|
| 规模 | 15 物体 · 约 **10 h** |
| 表示 | 物体几何 + 物体运动 + 人体全身姿态 |
| 代码 | <https://github.com/lijiaman/omomo_release> |
| 项目页 | <https://lijiaman.github.io/projects/omomo/> |
| 论文 | arXiv:[2309.16237](https://arxiv.org/abs/2309.16237) |

## 流程总览（机器人侧典型用法）

```mermaid
flowchart LR
  omomo[OMOMO 人体 HOI MoCap]
  ret[重定向引擎<br/>OmniRetarget / GMR / holosoma]
  ref[机器人参考轨迹]
  wbt[WBT / RL 跟踪训练]
  omomo --> ret --> ref --> wbt
```

## 常见误区或局限

- **不是机器人关节角**：与 [PHUMA](./dataset-bfm-phuma.md) 不同，OMOMO 主体为 **人体** 表示，上机前必须重定向并做动力学一致化。
- **依赖 SMPL 模型**：下载与可视化需注册 **SMPL-H / SMPL-X**；许可链条独立于代码仓库。
- **生成 vs 采集**：仓库同时含 **条件扩散合成** 代码；机器人研究通常消费 **已采集 MoCap 子集**，而非在线生成。

## 与其他页面的关系

- **下游重定向集**：[OmniRetarget 数据集](./omniretarget-dataset.md)（G1，`robot-object/` 来自 OMOMO）
- **重定向方法**：[OmniRetarget](./paper-hrl-stack-03-omniretarget.md)、[GMR](../methods/motion-retargeting-gmr.md)
- **对照阅读**：[humanoid-reference-motion-datasets 对比](../comparisons/humanoid-reference-motion-datasets.md)

## 参考来源

- [OMOMO 仓库归档](../../sources/repos/omomo_release.md)
- Li et al., *Object Motion Guided Human Motion Synthesis*, ACM TOG / SIGGRAPH Asia 2023
- GitHub：<https://github.com/lijiaman/omomo_release>

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [OmniRetarget 数据集](./omniretarget-dataset.md)
- [holosoma](./holosoma.md)
- [AMASS](./amass.md)
- [LaFAN1](./lafan1-dataset.md)

## 推荐继续阅读

- [OmniRetarget 项目页](https://omniretarget.github.io/) — OMOMO 物体交互重定向质量对比
- [ResMimic](./paper-resmimic.md) — AMASS + OMOMO 训练 GMT 先验的 loco-manipulation 范例
