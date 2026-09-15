---
type: entity
tags: [paper, humanoid, teleoperation, vr, mit]
status: complete
updated: 2026-09-15
arxiv: "2609.07933"
related:
  - ../tasks/teleoperation.md
  - ../tasks/loco-manipulation.md
  - ./paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md
sources:
  - ../../sources/papers/spot_humanoid_teleoperation_arxiv_2609_07933.md
summary: "SPOT（arXiv:2609.07933）：binocular fisheye + wide FOV stereo display; visual stabilization; viewpoint-action decoupling; operator head turn doesn；截至入库日未见官方代码。"
---

# SPOT（arXiv:2609.07933）

**SPOT**（*SPOT: Spatial Perception-Oriented Long-Horizon Humanoid Teleoperation*，[arXiv:2609.07933](https://arxiv.org/abs/2609.07933)）由 **马萨诸塞大学阿默斯特分校（UMass Amherst）；麻省理工（MIT）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

SPOT：面向空间感知的长时人形机器人遥操作 — binocular fisheye + wide FOV stereo display。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOV | Field of View | 视场角 |
| VR | Virtual Reality | 虚拟现实显示 |
| HRI | Human-Robot Interaction | 人机交互 |

## 为什么重要

传统第一人称遥操作把操作员头动直接映射机器人头，长时任务易晕且空间感知差。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 马萨诸塞大学阿默斯特分校（UMass Amherst）；麻省理工（MIT） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

机器人端双目鱼眼采集；操作员端广 FOV 立体显示 + 视觉稳定；viewpoint-action decoupling 分离观察视点与执行动作。

### 流程总览

```mermaid
flowchart LR
  fish[双目鱼眼] --> stream[立体视频流]
  stream --> stab[视觉稳定]
  stab --> hmd[操作员 HMD]
  hmd --> decouple[视点-动作解耦]
  decouple --> robot[人形执行]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 鱼眼标定与立体对齐；解耦映射需防止操作员迷失方向。 |

## 实验与评测

长时遥操作任务完成率、晕动评分、空间感知问卷。

## 结论

SPOT 通过视点-动作解耦与广 FOV 立体显示改善长时人形遥操作空间感知。

1. 双目鱼眼贴近机器人真实感知。
2. 视觉稳定减轻画面抖动。
3. 头动不驱动机器人头降低晕动。
4. 适合长时 loco-manipulation。
5. 硬件管线是系统核心贡献。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 头动映射遥操作 | 易晕、头颈负载大 |
| 第三人称遥操作 | 空间深度感知弱 |

## 局限与风险

依赖定制显示与相机；网络延迟未作为主线讨论。

## 关联页面

- [teleoperation](../tasks/teleoperation.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [./paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md](./paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md)

## 参考来源

- [spot_humanoid_teleoperation_arxiv_2609_07933.md](../../sources/papers/spot_humanoid_teleoperation_arxiv_2609_07933.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.07933](https://arxiv.org/abs/2609.07933)
